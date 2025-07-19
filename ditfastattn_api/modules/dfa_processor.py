import torch
from diffusers.models.attention_processor import Attention, AttnProcessor2_0, JointAttnProcessor2_0
from typing import List, Optional
import torch.nn.functional as F
import dfav2
import copy
from time import time
from ditfastattn_api.modules.ilp import solve_ip

# from natten.functional import na1d, na2d
import torch.nn as nn

from torch.nn.attention.flex_attention import _mask_mod_signature, or_masks
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
from functools import partial

def generate_sliding_window(window_size: int, block_size=64) -> _mask_mod_signature:
    """Generates a sliding window attention mask with a given window size.
    Args:
        window_size: The size of the sliding window.

    Note:
        We assume that the window size represents the lookback size and we mask out all future tokens
        similar to causal masking.
    """

    def sliding_window_with_offset(b, h, q_idx, kv_idx):
        return (torch.abs(q_idx - kv_idx) <= window_size // 2)

    sliding_window_mask = sliding_window_with_offset
    sliding_window_mask.__name__ = f"sliding_window_{window_size}"
    return sliding_window_mask

def generate_sliding_window_per_head(head_list: List, block_size=64) -> _mask_mod_signature:
    """Generates a sliding window attention mask with a given window size.
    Args:
        window_size: The size of the sliding window.

    Note:
        We assume that the window size represents the lookback size and we mask out all future tokens
        similar to causal masking.
    """

    def sliding_window_with_offset(b, h, q_idx, kv_idx):
        window_size = head_list[h]
        return (torch.abs(q_idx - kv_idx) <= window_size // 2) & (window_size > 0)

    sliding_window_mask = sliding_window_with_offset
    sliding_window_mask.__name__ = f"sliding_window_multiple_size"
    return sliding_window_mask

def headwise_cfg_attention(query, key, value, attn_func, head_mask, block_mask):
    """
    自定义 attention 计算：
    - 对于 head_mask 为 True 的 head，只计算前一半 batch 的 attention，并复制到后一半。
    - 对于 head_mask 为 False 的 head，正常计算所有 batch 的 attention。
    """
    B, H, S, D = query.size()
    half_B = B // 2

    # 分离需要替换和不需要替换的 head
    query_replace = query[:, head_mask, :, :]  # shape: (B, S, H_replace, D)
    key_replace = key[:, head_mask, :, :]
    value_replace = value[:, head_mask, :, :]

    query_normal = query[:, ~head_mask, :, :]  # shape: (B, S, H_normal, D)
    key_normal = key[:, ~head_mask, :, :]
    value_normal = value[:, ~head_mask, :, :]

    # 对于需要替换的 head，只计算前一半 batch 的 attention
    if query_replace.size(2) > 0:  # 如果有需要替换的 head
        query_first_half = query_replace[:half_B]  # shape: (half_B, S, H_replace, D)
        key_first_half = key_replace[:half_B]
        value_first_half = value_replace[:half_B]

        attention_first_half = attn_func(query_first_half, 
                                         key_first_half, 
                                         value_first_half, 
                                         block_mask=block_mask[1]).transpose(1,2)
        torch.cuda.synchronize()
        attention_second_half = attention_first_half.clone()  # 复制到后一半 batch
        attention_replace = torch.cat([attention_first_half, attention_second_half], dim=0)  # shape: (B, S, H_replace, D)
    else:
        attention_replace = torch.empty(B, S, 0, D, device=query.device)  # 如果没有需要替换的 head，返回空张量

    # 对于不需要替换的 head，正常计算所有 batch 的 attention
    if query_normal.size(2) > 0:  # 如果有不需要替换的 head
        attention_normal = attn_func(query_normal, 
                                     key_normal, 
                                     value_normal,
                                     block_mask=block_mask[0]).transpose(1,2)  # shape: (B, S, H_normal, D)
        torch.cuda.synchronize()
    else:
        attention_normal = torch.empty(B, S, 0, D, device=query.device)  # 如果没有不需要替换的 head，返回空张量

    # 合并两部分的结果
    attention_output = torch.cat([attention_replace, attention_normal], dim=2)  # shape: (B, S, H, D)

    # 恢复原始 head 的顺序
    head_indices = torch.arange(H, device=query.device)
    replace_indices = head_indices[head_mask]
    normal_indices = head_indices[~head_mask]
    original_order = torch.cat([replace_indices, normal_indices]).argsort()
    attention_output = attention_output[:, :, original_order, :]

    return attention_output

class DiTFastAttnProcessor:
    def __init__(self, steps_method=None, window_func=None, cond_first=None, alpha=0):
        self.steps_method = steps_method
        # CFG order flag (conditional first or unconditional first)
        self.cond_first = cond_first
        # Check at which timesteps do we need to compute the full-window residual
        # of this attention module
        self.window_func = window_func
        self.forward_mode = "normal"
        self.block_mask = {}
        self.prev_calib_output = None
        self.cached_output = None
        self.dfa_config=None
        self.stepi=0
        self.alpha = alpha
        self.wt = {}
        self.timestep_block_mask = {}
        self.asc_enable = False

    def raw_qkv_process_func(self, query, key, value, forward_args):
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False
        ).transpose(1,2)
        return hidden_states
    
    def calib_qkv_process_func(self, query, key, value, forward_args):
        # window_size_candidates = [0.995,0.99,0.98,0.97,0.96,0.95]
        headwise_relative_MSE={} # key (head, method) value: relative MSE = 均方误差（MSE）与目标变量方差的比值 （maybe）
        # latency_delta={}
        attn=forward_args["attn"]
        candidates = self.dfa_config.get_available_candidates(attn.name)
        B, H, S, _ = query.shape
        full_hidden_states = dfav2.flash_attn_func(query.transpose(1,2), key.transpose(1,2), value.transpose(1,2))
        for candidate in candidates:
            if "window_attn" in candidate:
                # without residual share
                window_size_factor = int(candidate.split("_")[-1])
                if window_size_factor not in self.block_mask.keys():
                    sliding_window_mask = generate_sliding_window(window_size=S // window_size_factor)
                    block_mask = create_block_mask(sliding_window_mask, None, None, S, S, device="cuda", _compile=True, BLOCK_SIZE=128)
                    self.block_mask[window_size_factor] = block_mask
                hidden_states = self.window_func(
                    query, key, value, block_mask=self.block_mask[window_size_factor]
                ).transpose(1, 2)
                torch.cuda.synchronize()
                # with residual share (latency *1.5)
            # output share
            elif candidate == 'output_share':
                hidden_states = self.prev_calib_output

            # add cfg sharing
            if "cfg_share" in candidate:
                hidden_states = hidden_states[:B//2].repeat(2, 1, 1, 1)

            # breakpoint()
            rse = (((hidden_states - full_hidden_states)**2).mean(dim=(0,1,3)) / ((full_hidden_states - full_hidden_states.mean(dim=(0,1,3), keepdim=True))**2).mean(dim=(0,1,3))).detach().cpu()
            # print(rse)
            for h in range(rse.size()[0]):
                headwise_relative_MSE[(h, candidate)] = rse[h].item()

        # ILP: target is to minimize the latency given MSE is below a threshold (?)
        head_method_list = solve_ip(headwise_relative_MSE,self.dfa_config.latency, self.alpha)
        print(head_method_list)
        if self.stepi not in self.dfa_config.plan.keys():
            self.dfa_config.plan[self.stepi] = {}
        self.dfa_config.plan[self.stepi][attn.name] = head_method_list

        # breakpoint()
        # per head, gen kernel
        if "cfg_share" not in candidates:
            wt = torch.ones(16, dtype=torch.int64) * (S * 2)
            self.output_share_dict[self.stepi] = []
            for head_method in head_method_list:
                head = head_method[0]
                ws = head_method[1]
                if ws == "0":
                    wt[head] = 0
                    self.output_share_dict[self.stepi].append(head)
                else:
                    wt[head] = S // (int(ws) * 2)
            self.wt[self.stepi] = wt

            if self.stepi not in self.timestep_block_mask.keys():
                print(f"create block mask for {self.stepi}")
                sliding_window_per_head_mask = generate_sliding_window_per_head(wt.to(query.device))
                block_mask = create_block_mask(sliding_window_per_head_mask, None, H, S, S, device="cuda", _compile=True, BLOCK_SIZE=128)
            # if attn.name == "transformer_blocks.7.attn":
            #     breakpoint()
                self.timestep_block_mask[self.stepi] = block_mask
            output = self.window_func(
                query, key, value, block_mask=self.timestep_block_mask[self.stepi]
            ).transpose(1, 2)
            torch.cuda.synchronize()

        else:
            self.asc_enable = True
            wt = torch.ones(16, dtype=torch.int64) * (S * 2)
            cfg_mask = [False] * 16
            cfg_mask = torch.tensor(cfg_mask)
            self.output_share_dict[self.stepi] = []
            for head_method in head_method_list:
                head = head_method[0]
                ws = head_method[1]
                if int(ws) < 0:
                    cfg_mask[head] = True
                if ws == "0":
                    wt[head] = 0
                    self.output_share_dict[self.stepi].append(head)
                else:
                    wt[head] = (S - 333) // abs(int(ws))
            self.wt[self.stepi] = wt
            if self.cfg_mask is None:
                self.cfg_mask = {}
            self.cfg_mask[self.stepi] = cfg_mask
            if self.stepi not in self.timestep_block_mask.keys():
                print(f"create block mask for {self.stepi}")
                if any(cfg_mask):
                    print("asc adopted")
                    sliding_window_per_head_mask_asc = generate_sliding_window_per_head(wt.to(query.device)[cfg_mask])
                    block_mask_asc = create_block_mask(sliding_window_per_head_mask_asc, None, cfg_mask.sum().item(), S, S, device="cuda", _compile=True, BLOCK_SIZE=128)
                else:
                    print("asc not adopted")
                    block_mask_asc = None
                sliding_window_per_head_mask = generate_sliding_window_per_head(wt.to(query.device)[~cfg_mask])
                block_mask = create_block_mask(sliding_window_per_head_mask, None, H - cfg_mask.sum().item(), S, S, device="cuda", _compile=True, BLOCK_SIZE=128)
                self.timestep_block_mask[self.stepi] = (block_mask, block_mask_asc)
            
            if block_mask_asc is not None:
                output = headwise_cfg_attention(query, key, value, self.window_func, cfg_mask, self.timestep_block_mask[self.stepi])
            else:
                output = self.window_func(
                    query, key, value, block_mask=self.timestep_block_mask[self.stepi][0]
                ).transpose(1, 2)
        # B S H D
        output[:,:,self.output_share_dict[self.stepi],:] = self.prev_calib_output[:,:,self.output_share_dict[self.stepi],:]
        return output
    

    def qkv_process_perhead_func(self, query, key, value, forward_args):
        breakpoint()
        _, _, S, _ = query.shape
        
        if self.asc_enable:
            if self.timestep_block_mask[self.stepi][1] is not None:
                output = headwise_cfg_attention(query, key, value, self.window_func, self.cfg_mask[self.stepi], self.timestep_block_mask[self.stepi])
            else:
                output = self.window_func(
                    query, key, value, block_mask=self.timestep_block_mask[self.stepi][0]
                ).transpose(1, 2)
        else:
            output = self.window_func(
                query, key, value, block_mask=self.timestep_block_mask[self.stepi]
            ).transpose(1, 2)
            # self.window_func(query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2), block_mask=self.timestep_block_mask[self.stepi]).transpose(1, 2)
            torch.cuda.synchronize()
        if output[:,:,self.output_share_dict[self.stepi],:].shape != self.cached_output.shape:
            breakpoint()
        output[:,:,self.output_share_dict[self.stepi],:] = self.cached_output
        if self.stepi+1 in self.output_share_dict.keys():
            self.cached_output = output[:,:,self.output_share_dict[self.stepi+1],:]
        return output
    
    def raw_qkv_process_after_calib_func(self, query, key, value, forward_args):
        hidden_states = dfav2.flash_attn_func(query.transpose(1,2), key.transpose(1,2), value.transpose(1,2))
        if self.stepi+1 in self.output_share_dict.keys():
            self.cached_output = hidden_states[:,:,self.output_share_dict[self.stepi+1],:]
        return hidden_states
    

    def run_forward(self, attn:Attention, hidden_states, encoder_hidden_states, qkv_process_func, temb):
        forward_args = dict(attn=attn, hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states)
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = qkv_process_func(
            query, key, value, forward_args
        )

        if self.forward_mode == "calib_collect_info":
            self.prev_calib_output = hidden_states.detach()

        hidden_states = hidden_states.reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
        *args,
        **kwargs,
    ) -> torch.FloatTensor:
        if hasattr(attn, "qkv"):
            hidden_states = self.run_opensora_forward(
                attn,
                hidden_states,
                encoder_hidden_states,
                attention_mask,
                temb,
                self.steps_method[self.stepi],
            )
        else:
            if self.forward_mode == "calib_collect_info":
                if self.stepi != 0:
                # qkv_process_func=self.calib_collect_info_qkv_process_func
                    qkv_process_func=  self.calib_qkv_process_func
                else:
                    qkv_process_func=self.raw_qkv_process_func
        
            elif self.forward_mode == "normal":
                qkv_process_func=self.raw_qkv_process_func

            elif self.forward_mode == "perhead_normal":
                # print(self.stepi)
                if self.stepi == 0:
                    # qkv_process_func = self.raw_qkv_process_after_calib_func
                    qkv_process_func = self.raw_qkv_process_after_calib_func
                else:
                    # qkv_process_func = self.qkv_process_perhead_func
                    qkv_process_func = self.qkv_process_perhead_func
            hidden_states = self.run_forward(
                attn,
                hidden_states,
                encoder_hidden_states,
                qkv_process_func,
                temb
            )
        # After been call once, add the timestep index of this attention module by 1
        return hidden_states
    

    # def update_config(self, **kwargs):
    #     for key, value in kwargs.items():
    #         setattr(self, key, copy.deepcopy(value))
    #     self.raw_steps_residual_config = self.compute_raw_steps_residual_config(self.steps_method)

    # def compute_raw_steps_residual_config(self, steps_method):
    #     steps_residual_config = []
    #     # assert steps_method[0] == "raw", "The first step of DiTFastAttnProcessor must be raw"
    #     for i, method in enumerate(steps_method):
    #         residual_config = (False, None)
    #         if "raw" in method:
    #             for j in range(i + 1, len(steps_method)):
    #                 if "residual_window_attn" in steps_method[j] and "without_residual" not in steps_method[j]:
    #                     # If encountered a step that conduct WA-RS,
    #                     # this step needs the residual computation
    #                     window_size = int(steps_method[j].split("_")[-1])
    #                     residual_config = (True, (window_size // 2, window_size // 2))
    #                     break
    #                 if "raw" in steps_method[j]:
    #                     # If encountered another step using the `full-attn` strategy,
    #                     # this step doesn't need the residual computation
    #                     break
    #         steps_residual_config.append(residual_config)
    #     return steps_residual_config

    # def run_forward_method(self, m, hidden_states, encoder_hidden_states, attention_mask, temb, method):
    #     residual = hidden_states
    #     if method == "output_share":
    #         hidden_states = self.cached_output
    #     else:
    #         if "cfg_attn_share" in method:
    #             # Directly use the unconditional branch's attention output
    #             # as the conditional branch's attention output

    #             # TODO: Maybe use the conditional branch's attention output
    #             # as the unconditional's is better
    #             batch_size = hidden_states.shape[0]
    #             if self.cond_first:
    #                 hidden_states = hidden_states[: batch_size // 2]
    #             else:
    #                 hidden_states = hidden_states[batch_size // 2 :]
    #             if encoder_hidden_states is not None:
    #                 if self.cond_first:
    #                     encoder_hidden_states = encoder_hidden_states[: batch_size // 2]
    #                 else:
    #                     encoder_hidden_states = encoder_hidden_states[batch_size // 2 :]
    #             if attention_mask is not None:
    #                 if self.cond_first:
    #                     attention_mask = attention_mask[: batch_size // 2]
    #                 else:
    #                     attention_mask = attention_mask[batch_size // 2 :]

    #         if m.spatial_norm is not None:
    #             hidden_states = m.spatial_norm(hidden_states, temb)

    #         input_ndim = hidden_states.ndim

    #         if input_ndim == 4:
    #             batch_size, channel, height, width = hidden_states.shape
    #             hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

    #         batch_size, sequence_length, _ = (
    #             hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
    #         )
    #         if attention_mask is not None:
    #             attention_mask = m.prepare_attention_mask(attention_mask, sequence_length, batch_size)
    #             # scaled_dot_product_attention expects attention_mask shape to be
    #             # (batch, heads, source_length, target_length)
    #             attention_mask = attention_mask.view(batch_size, m.heads, -1, attention_mask.shape[-1])

    #         if m.group_norm is not None:
    #             hidden_states = m.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

    #         query = m.to_q(hidden_states)

    #         if encoder_hidden_states is None:
    #             encoder_hidden_states = hidden_states
    #         elif m.norm_cross:
    #             encoder_hidden_states = m.norm_encoder_hidden_states(encoder_hidden_states)

    #         key = m.to_k(encoder_hidden_states)
    #         value = m.to_v(encoder_hidden_states)

    #         inner_dim = key.shape[-1]
    #         head_dim = inner_dim // m.heads

    #         query = query.view(batch_size, -1, m.heads, head_dim)

    #         key = key.view(batch_size, -1, m.heads, head_dim)
    #         value = value.view(batch_size, -1, m.heads, head_dim)

    #         if attention_mask is not None:
    #             assert "residual_window_attn" not in method

    #             hidden_states = F.scaled_dot_product_attention(
    #                 query.transpose(1, 2),
    #                 key.transpose(1, 2),
    #                 value.transpose(1, 2),
    #                 attn_mask=attention_mask,
    #                 dropout_p=0.0,
    #                 is_causal=False,
    #             ).transpose(1, 2)
    #         elif "raw" in method:
    #             all_hidden_states = dfav2.flash_attn_func(query, key, value)
    #             if self.raw_steps_residual_config[self.stepi][0] == True:
    #                 # Compute the full-window attention residual
    #                 window_size = self.raw_steps_residual_config[self.stepi][1]
    #                 w_hidden_states = dfav2.flash_attn_func(query, key, value, window_size=window_size)
    #                 w_residual = all_hidden_states - w_hidden_states
    #                 if "cfg_attn_share" in method:
    #                     w_residual = torch.cat([w_residual, w_residual], dim=0)
    #                 # Save the residual for usage in follow-up steps
    #                 m.cached_residual = w_residual
    #             elif self.cache_residual_forced:
    #                 # Compute the full-window attention residual
    #                 window_size = [64, 64]
    #                 w_hidden_states = dfav2.flash_attn_func(query, key, value, window_size=window_size)
    #                 w_residual = all_hidden_states - w_hidden_states
    #                 if "cfg_attn_share" in method:
    #                     w_residual = torch.cat([w_residual, w_residual], dim=0)
    #                 # Save the residual for usage in follow-up steps
    #                 m.cached_residual = w_residual
    #             hidden_states = all_hidden_states
    #         elif "residual_window_attn" in method:
    #             window_size = int(method.split("_")[-1])
    #             w_hidden_states = dfav2.flash_attn_func(
    #                 query, key, value, window_size=(window_size // 2, window_size // 2)
    #             )

    #             if "without_residual" in method:
    #                 # For ablation study of `residual_window_attn+without_residual`
    #                 # v.s. `residual_window_attn`
    #                 hidden_states = w_hidden_states
    #             else:
    #                 hidden_states = w_hidden_states + m.cached_residual[:batch_size].view_as(w_hidden_states)

    #         hidden_states = hidden_states.reshape(batch_size, -1, m.heads * head_dim)
    #         hidden_states = hidden_states.to(query.dtype)


    #         # linear proj
    #         hidden_states = m.to_out[0](hidden_states)
    #         # dropout
    #         hidden_states = m.to_out[1](hidden_states)

    #         if input_ndim == 4:
    #             hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

    #         if "cfg_attn_share" in method:
    #             hidden_states = torch.cat([hidden_states, hidden_states], dim=0)

    #         if self.need_cache_output:
    #             self.cached_output = hidden_states

    #     if m.residual_connection:
    #         hidden_states = hidden_states + residual
    #     hidden_states = hidden_states / m.rescale_output_factor

    #     return hidden_states

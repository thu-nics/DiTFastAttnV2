import torch
from diffusers.models.attention_processor import Attention, AttnProcessor2_0, JointAttnProcessor2_0
from typing import List, Optional
import torch.nn.functional as F
# import flash_attn
from dfav2 import flash_attn_func
import copy
from time import time
from ditfastattn_api.modules.ilp import solve_ip

# from natten.functional import na1d, na2d
import torch.nn as nn

from torch.nn.attention.flex_attention import _mask_mod_signature, or_masks
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
from functools import partial
from ditfastattn_api.video_reorder import sparse_head_placement, hidden_states_placement

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
        return ((torch.abs(q_idx - kv_idx) <= window_size // 2)) & (window_size > 0)

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

class WanFastAttnProcessor:
    def __init__(self, steps_method=None, cond_first=None, window_func=None, alpha=0):
        self.steps_method = steps_method
        self.cond_first = cond_first
        self.window_func = window_func
        self.forward_mode = "normal" # could be "normal" or "calib_get_grad" or "calib_post_inference"
        self.block_mask = {}
        self.prev_calib_output = None
        self.cached_output = None
        self.dfa_config=None
        self.stepi=0
        self.alpha = alpha
        self.wt = {}
        self.timestep_block_mask = {}
        
        # self.attn_weight=0
        # self.attn_weight_num_count=0
        self.relative_MSE_threshold=0.01 # hyperparameter
        self.evaluated_latency=None
        self.output_share_dict = {}
        self.cfg_mask = None
        self.asc_enable = False

        self.num_frames = None
        self.height = None
        self.width = None
        
        self.reorder_mask = {}
        
    
    def run_forward(self, attn:Attention, hidden_states, encoder_hidden_states, attention_mask, rotary_emb, qkv_process_func):
        
        forward_args = dict(attn=attn, hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states)
        
        encoder_hidden_states_img = None
        if attn.add_k_proj is not None:
            # 512 is the context length of the text encoder, hardcoded for now
            image_context_length = encoder_hidden_states.shape[1] - 512
            encoder_hidden_states_img = encoder_hidden_states[:, :image_context_length]
            encoder_hidden_states = encoder_hidden_states[:, image_context_length:]
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

        if rotary_emb is not None:

            def apply_rotary_emb(hidden_states: torch.Tensor, freqs: torch.Tensor):
                dtype = torch.float32 if hidden_states.device.type == "mps" else torch.float64
                x_rotated = torch.view_as_complex(hidden_states.to(dtype).unflatten(3, (-1, 2)))
                x_out = torch.view_as_real(x_rotated * freqs).flatten(3, 4)
                return x_out.type_as(hidden_states)

            query = apply_rotary_emb(query, rotary_emb)
            key = apply_rotary_emb(key, rotary_emb)

        # I2V task
        hidden_states_img = None
        if encoder_hidden_states_img is not None:
            key_img = attn.add_k_proj(encoder_hidden_states_img)
            key_img = attn.norm_added_k(key_img)
            value_img = attn.add_v_proj(encoder_hidden_states_img)

            key_img = key_img.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            value_img = value_img.unflatten(2, (attn.heads, -1)).transpose(1, 2)

            hidden_states_img = F.scaled_dot_product_attention(
                query, key_img, value_img, attn_mask=None, dropout_p=0.0, is_causal=False
            )
            hidden_states_img = hidden_states_img.transpose(1, 2).flatten(2, 3)
            hidden_states_img = hidden_states_img.type_as(query)


        hidden_states = qkv_process_func(query, key, value, forward_args)
        if self.forward_mode == "calib_collect_info":
            self.prev_calib_output = hidden_states.detach()
        
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.type_as(query)

        if hidden_states_img is not None:
            hidden_states = hidden_states + hidden_states_img

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states

    
    def calib_qkv_process_func(self, query, key, value, forward_args):
        # window_size_candidates = [0.995,0.99,0.98,0.97,0.96,0.95]
        headwise_relative_MSE={} # key (head, method) value: relative MSE = 均方误差（MSE）与目标变量方差的比值 （maybe）
        # latency_delta={}
        attn=forward_args["attn"]
        candidates = self.dfa_config.get_available_candidates(attn.name)
        B, H, S, _ = query.shape
        full_hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False
        ).transpose(1,2)
        for candidate in candidates:
            if "arrow_attn" in candidate:
                if "reorder" in candidate:
                    output_hidden_states = torch.zeros_like(query)
                    query_out, key_out, value_out = torch.zeros_like(query), torch.zeros_like(key), torch.zeros_like(value)
                    is_reorder = torch.ones((B,H), device=query.device, dtype=torch.int32)
                    sparse_head_placement(query, key, value, 
                                            query_out, key_out, value_out, 
                                            is_reorder, 
                                            0, 
                                            self.num_frames, 
                                            self.height * self.width)
                    window_size_factor = int(candidate.split("_")[-1])
                    if window_size_factor not in self.block_mask.keys():
                        sliding_window_mask = generate_sliding_window(window_size= (S // window_size_factor))
                        block_mask = create_block_mask(sliding_window_mask, None, None, S, S, device="cuda", _compile=True, BLOCK_SIZE=128)
                        self.block_mask[window_size_factor] = block_mask
                    hidden_states = self.window_func(
                        query_out, key_out, value_out, block_mask=self.block_mask[window_size_factor]
                    )
                    torch.cuda.synchronize()

                    hidden_states_placement(hidden_states, 
                                            output_hidden_states, 
                                            is_reorder, 
                                            0, 
                                            self.num_frames, 
                                            self.height * self.width)
                    hidden_states = output_hidden_states.transpose(1, 2)
                else:
                    # without residual share
                    window_size_factor = int(candidate.split("_")[-1])
                    if window_size_factor not in self.block_mask.keys():
                        sliding_window_mask = generate_sliding_window(window_size= (S // window_size_factor))
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

            rse = (((hidden_states - full_hidden_states)**2).mean(dim=(0,1,3)) / ((full_hidden_states - full_hidden_states.mean(dim=(0,1,3), keepdim=True))**2).mean(dim=(0,1,3))).detach().cpu()
            for h in range(rse.size()[0]):
                if rse[h].item() < self.alpha / H * 1:
                    headwise_relative_MSE[(h, candidate)] = rse[h].item()

        # cache compression plan
        if self.stepi not in self.dfa_config.plan.keys():
            self.dfa_config.plan[self.stepi] = {}
        if bool(headwise_relative_MSE):
            head_method_list = solve_ip(headwise_relative_MSE,self.dfa_config.latency, self.alpha)
            print(head_method_list)
        else:
            head_method_list = {}

        self.dfa_config.plan[self.stepi][attn.name] = head_method_list

        # per head, gen kernel
        if "cfg_share" not in candidates:
            wt = torch.ones(H, dtype=torch.int64) * (S * 2)
            reorder_mask = torch.zeros(H, dtype=torch.int64, device=query.device)
            self.output_share_dict[self.stepi] = []
            for head_method in head_method_list:
                head = head_method[0]
                ws = head_method[1]
                if ws == "0":
                    wt[head] = 0
                    self.output_share_dict[self.stepi].append(head)
                else:
                    if ws.startswith("re"):
                        ws = ws.strip("re")
                        reorder_mask[head] = 1
                    wt[head] = S // int(ws)
            self.wt[self.stepi] = wt
            self.reorder_mask[self.stepi] = reorder_mask

            output_hidden_states = torch.zeros_like(query)
            query_out, key_out, value_out = torch.zeros_like(query), torch.zeros_like(key), torch.zeros_like(value)
            sparse_head_placement(query, key, value, 
                                    query_out, key_out, value_out,
                                    self.reorder_mask[self.stepi].repeat(B, 1),
                                    0,
                                    self.num_frames,
                                    self.height * self.width)

            if self.stepi not in self.timestep_block_mask.keys():
                print(f"create block mask for {self.stepi}")
                sliding_window_per_head_mask = generate_sliding_window_per_head(wt.to(query.device))
                block_mask = create_block_mask(sliding_window_per_head_mask, None, H, S, S, device="cuda", _compile=True, BLOCK_SIZE=128)
                self.timestep_block_mask[self.stepi] = block_mask
            output = self.window_func(
                query_out, key_out, value_out, block_mask=self.timestep_block_mask[self.stepi]
            )
            torch.cuda.synchronize()

            hidden_states_placement(output, 
                                output_hidden_states, 
                                self.reorder_mask[self.stepi].repeat(B, 1),
                                0, 
                                self.num_frames, 
                                self.height * self.width)
            output = output_hidden_states.transpose(1,2)

        else:
            self.asc_enable = True
            wt = torch.ones(H, dtype=torch.int64) * (S * 2)
            reorder_mask = torch.zeros(H, dtype=torch.int64, device=query.device)
            cfg_mask = [False] * H
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
                    if ws.startswith("re"):
                        ws = ws.strip("re")
                        self.reorder_dict[self.stepi].append(head)
                    wt[head] = S // abs(int(ws))
            self.wt[self.stepi] = wt
            self.reorder_mask[self.stepi] = reorder_mask
            if self.cfg_mask is None:
                self.cfg_mask = {}
            self.cfg_mask[self.stepi] = cfg_mask

            output_hidden_states = torch.zeros_like(query)
            query_out, key_out, value_out = torch.zeros_like(query), torch.zeros_like(key), torch.zeros_like(value)
            sparse_head_placement(query, key, value, 
                                    query_out, key_out, value_out,
                                    self.reorder_mask[self.stepi].repeat(B, 1),
                                    0,
                                    self.num_frames,
                                    self.height * self.width)

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
                output = headwise_cfg_attention(query_out, key_out, value_out, self.window_func, cfg_mask, self.timestep_block_mask[self.stepi]).transpose(1,2)
            else:
                output = self.window_func(
                    query_out, key_out, value_out, block_mask=self.timestep_block_mask[self.stepi][0]
                )

            hidden_states_placement(output, 
                                output_hidden_states, 
                                self.reorder_mask[self.stepi].repeat(B, 1),
                                0, 
                                self.num_frames, 
                                self.height * self.width)
            output = output_hidden_states.transpose(1,2)

        # B S H D
        output[:,:,self.output_share_dict[self.stepi],:] = self.prev_calib_output[:,:,self.output_share_dict[self.stepi],:]
        return output
    
    def raw_qkv_process_func(self, query, key, value, forward_args):
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False
        ).transpose(1,2)
        return hidden_states
    
    
    def qkv_process_perhead_func(self, query, key, value, forward_args):
        # breakpoint()
        B, _, S, _ = query.shape
    
        output_hidden_states = torch.zeros_like(query)
        query_out, key_out, value_out = torch.zeros_like(query), torch.zeros_like(key), torch.zeros_like(value)
        sparse_head_placement(query, key, value, 
                                query_out, key_out, value_out,
                                self.reorder_mask[self.stepi].repeat(B, 1),
                                0,
                                self.num_frames,
                                self.height * self.width)

        if self.asc_enable:
            if self.timestep_block_mask[self.stepi][1] is not None:
                output = headwise_cfg_attention(query_out, key_out, value_out, self.window_func, self.cfg_mask[self.stepi], self.timestep_block_mask[self.stepi])
            else:
                output = self.window_func(
                    query_out, key_out, value_out, block_mask=self.timestep_block_mask[self.stepi][0]
                )
        else:
            output = self.window_func(
                query_out, key_out, value_out, block_mask=self.timestep_block_mask[self.stepi]
            )
            # self.window_func(query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2), block_mask=self.timestep_block_mask[self.stepi]).transpose(1, 2)
            torch.cuda.synchronize()

        hidden_states_placement(output, 
                                output_hidden_states, 
                                self.reorder_mask[self.stepi].repeat(B, 1),
                                0, 
                                self.num_frames, 
                                self.height * self.width)
        output = output_hidden_states.transpose(1,2)
        
        if output[:,:,self.output_share_dict[self.stepi],:].shape != self.cached_output.shape:
            breakpoint()
        output[:,:,self.output_share_dict[self.stepi],:] = self.cached_output
        if self.stepi+1 in self.output_share_dict.keys():
            self.cached_output = output[:,:,self.output_share_dict[self.stepi+1],:]
        return output
    
    def raw_qkv_process_after_calib_func(self, query, key, value, forward_args):
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False
        ).transpose(1,2)
        if self.stepi+1 in self.output_share_dict.keys():
            self.cached_output = hidden_states[:,:,self.output_share_dict[self.stepi+1],:]
        return hidden_states
            
    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        rotary_emb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.FloatTensor:
        
        if self.forward_mode == "calib_collect_info":
            if self.stepi != 0:
                qkv_process_func=  self.calib_qkv_process_func
            else:
                qkv_process_func=self.raw_qkv_process_func
        
        elif self.forward_mode == "normal":
            qkv_process_func=self.raw_qkv_process_func

        elif self.forward_mode == "perhead_normal":
            if self.stepi == 0:
                qkv_process_func = self.raw_qkv_process_after_calib_func
            else:
                qkv_process_func = self.qkv_process_perhead_func
        
        hidden_states = self.run_forward(attn, hidden_states, encoder_hidden_states, attention_mask, rotary_emb, qkv_process_func)
        return hidden_states
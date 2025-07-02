import torch
from diffusers.models.attention_processor import Attention, AttnProcessor2_0, JointAttnProcessor2_0
from typing import List, Optional
import torch.nn.functional as F
import flash_attn
import copy
from time import time
from ditfastattn_api.modules.ilp import solve_ip

# from natten.functional import na1d, na2d
import torch.nn as nn

from torch.nn.attention.flex_attention import _mask_mod_signature, or_masks
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
from functools import partial

def generate_partial_sliding_window(window_size: int, vtok_len: int, block_size=64) -> _mask_mod_signature:
    """Generates a sliding window attention mask with a given window size.
    Args:
        window_size: The size of the sliding window.

    Note:
        We assume that the window size represents the lookback size and we mask out all future tokens
        similar to causal masking.
    """

    def sliding_window_with_offset(b, h, q_idx, kv_idx):
        return (torch.abs(q_idx - kv_idx) <= window_size // 2) | (q_idx >= vtok_len) | (kv_idx >= vtok_len)

    sliding_window_mask = sliding_window_with_offset
    sliding_window_mask.__name__ = f"sliding_window_{window_size}"
    return sliding_window_mask

def generate_partial_sliding_window_per_head(head_list: List, vtok_len: int, block_size=64) -> _mask_mod_signature:
    """Generates a sliding window attention mask with a given window size.
    Args:
        window_size: The size of the sliding window.

    Note:
        We assume that the window size represents the lookback size and we mask out all future tokens
        similar to causal masking.
    """

    def sliding_window_with_offset(b, h, q_idx, kv_idx):
        window_size = head_list[h]
        return ((torch.abs(q_idx - kv_idx) <= window_size // 2) | (q_idx >= vtok_len) | (kv_idx >= vtok_len)) & (window_size > 0)

    sliding_window_mask = sliding_window_with_offset
    sliding_window_mask.__name__ = f"sliding_window_multiple_size"
    return sliding_window_mask

def headwise_cfg_attention(query, key, value, attn_func, head_mask, block_mask):
    """
    自定义 attention 计算：
    - 对于 head_mask 为 True 的 head，只计算前一半 batch 的 attention，并复制到后一半。
    - 对于 head_mask 为 False 的 head，正常计算所有 batch 的 attention。
    """
    B, S, H, D = query.size()
    half_B = B // 2

    # 分离需要替换和不需要替换的 head
    query_replace = query[:, :, head_mask, :]  # shape: (B, S, H_replace, D)
    key_replace = key[:, :, head_mask, :]
    value_replace = value[:, :, head_mask, :]

    query_normal = query[:, :, ~head_mask, :]  # shape: (B, S, H_normal, D)
    key_normal = key[:, :, ~head_mask, :]
    value_normal = value[:, :, ~head_mask, :]

    # 对于需要替换的 head，只计算前一半 batch 的 attention
    if query_replace.size(2) > 0:  # 如果有需要替换的 head
        query_first_half = query_replace[:half_B]  # shape: (half_B, S, H_replace, D)
        key_first_half = key_replace[:half_B]
        value_first_half = value_replace[:half_B]

        attention_first_half = attn_func(query_first_half.transpose(1,2), 
                                         key_first_half.transpose(1,2), 
                                         value_first_half.transpose(1,2), 
                                         block_mask=block_mask[1]).transpose(1,2)
        torch.cuda.synchronize()
        attention_second_half = attention_first_half.clone()  # 复制到后一半 batch
        attention_replace = torch.cat([attention_first_half, attention_second_half], dim=0)  # shape: (B, S, H_replace, D)
    else:
        attention_replace = torch.empty(B, S, 0, D, device=query.device)  # 如果没有需要替换的 head，返回空张量

    # 对于不需要替换的 head，正常计算所有 batch 的 attention
    if query_normal.size(2) > 0:  # 如果有不需要替换的 head
        attention_normal = attn_func(query_normal.transpose(1,2), 
                                     key_normal.transpose(1,2), 
                                     value_normal.transpose(1,2),
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

class CogVideoXFastAttnProcessor:
    def __init__(self, steps_method=None, cond_first=None, window_func=None, alpha=0):
        self.steps_method = steps_method
        self.cond_first = cond_first
        self.window_func = window_func
        self.forward_mode = "normal" # could be "normal" or "calib_get_grad" or "calib_post_inference"
        self.raw_steps_residual_config = self.compute_raw_steps_residual_config(steps_method)
        self.block_mask = {}
        self.compression_influences = {}
        self.prev_calib_output = None
        self.cached_residual = {}
        self.cached_output = None
        self.dfa_config=None
        self.stepi=0
        self.curr_window_factor = None
        self.alpha = alpha
        self.wt = {}
        self.timestep_block_mask = {}
        
        self.relative_MSE_threshold=0.01 # hyperparameter
        self.evaluated_latency=None
        self.output_share_dict = {}
        self.cfg_mask = None
        self.asc_enable = False


    def create_attn_mask(window_size, seq_len, vtok_len):
        mask = torch.zeros(seq_len, seq_len)

        # Fill the mask with 1s within the window around each token
        for i in range(seq_len):
            # Calculate the start and end of the window
            if i < vtok_len:
                start = max(0, i - window_size)
                end = min(vtok_len, i + window_size + 1)
                mask[i, start:end] = 1
                mask[i,vtok_len:] = 1
            else:
                mask[i, :] = 1
        return mask
        
        
    def update_config(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, copy.deepcopy(value))
        self.raw_steps_residual_config = self.compute_raw_steps_residual_config(self.steps_method)
        
    def compute_raw_steps_residual_config(self, steps_method):
        steps_residual_config = []
        # assert steps_method[0] == "raw", "The first step of DiTFastAttnProcessor must be raw"
        for i, method in enumerate(steps_method):
            residual_config = (False, None)
            if "raw" in method:
                for j in range(i + 1, len(steps_method)):
                    if "residual_window_attn" in steps_method[j] and "without_residual" not in steps_method[j]:
                        # If encountered a step that conduct WA-RS,
                        # this step needs the residual computation
                        window_size = int(steps_method[j].split("_")[-1])
                        residual_config = (True, (window_size // 2, window_size // 2))
                        break
                    if "raw" in steps_method[j]:
                        # If encountered another step using the `full-attn` strategy,
                        # this step doesn't need the residual computation
                        break
            steps_residual_config.append(residual_config)
        return steps_residual_config
    
    def get_attention_scores(
        self, query: torch.Tensor, key: torch.Tensor, scale, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        r"""
        Compute the attention scores.

        Args:
            query (`torch.Tensor`): The query tensor.
            key (`torch.Tensor`): The key tensor.
            attention_mask (`torch.Tensor`, *optional*): The attention mask to use. If `None`, no mask is applied.

        Returns:
            `torch.Tensor`: The attention probabilities/scores.
        """
        dtype = query.dtype
        if self.upcast_attention:
            query = query.float()
            key = key.float()

        # if attention_mask is None:
        #     baddbmm_input = torch.empty(
        #         query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device
        #     )
        #     beta = 0
        # else:
        #     baddbmm_input = attention_mask
        #     beta = 1

        baddbmm_input = torch.empty(
            query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device
        )
        beta = 0

        attention_scores = torch.baddbmm(
            baddbmm_input,
            query,
            key.transpose(-1, -2),
            beta=beta,
            alpha=scale,
        )
        del baddbmm_input

        # apply attention mask
        attention_scores = attention_scores * attention_mask.unsqueeze(0).unsqueeze(-1)

        if self.upcast_softmax:
            attention_scores = attention_scores.float()

        attention_probs = attention_scores.softmax(dim=-1)
        del attention_scores

        attention_probs = attention_probs.to(dtype)

        return attention_probs
    
    def run_forward(self, attn:Attention, hidden_states, encoder_hidden_states, qkv_process_func, image_rotary_emb):
        
        forward_args = dict(attn=attn, hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states)
        
        text_seq_length = encoder_hidden_states.size(1)

        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        # if attention_mask is not None:
        #     attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        #     attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Apply RoPE if needed
        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb

            query[:, :, text_seq_length:] = apply_rotary_emb(query[:, :, text_seq_length:], image_rotary_emb)
            if not attn.is_cross_attention:
                key[:, :, text_seq_length:] = apply_rotary_emb(key[:, :, text_seq_length:], image_rotary_emb)

        # attn
        hidden_states = qkv_process_func(query.transpose(1,2), key.transpose(1,2), value.transpose(1,2), forward_args)
        if self.forward_mode == "calib_collect_info":
            self.prev_calib_output = hidden_states.detach()
        
        hidden_states = hidden_states.reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        encoder_hidden_states, hidden_states = hidden_states.split(
            [text_seq_length, hidden_states.size(1) - text_seq_length], dim=1
        )
        return hidden_states, encoder_hidden_states
    
    def calib_collect_info_qkv_process_func(self, query, key, value, forward_args):
        # witout residual share window attention
        pass
    
    def calib_qkv_process_func(self, query, key, value, forward_args):
        # window_size_candidates = [0.995,0.99,0.98,0.97,0.96,0.95]
        headwise_relative_MSE={} # key (head, method) value: relative MSE = 均方误差（MSE）与目标变量方差的比值 （maybe）
        # latency_delta={}
        attn=forward_args["attn"]
        candidates = self.dfa_config.get_available_candidates(attn.name)
        B, S, H, _ = query.shape
        full_hidden_states = flash_attn.flash_attn_func(query, key, value)
        for candidate in candidates:
            if "window_attn" in candidate:
                # without residual share
                window_size_factor = int(candidate.split("_")[-1])
                if window_size_factor not in self.block_mask.keys():
                    sliding_window_mask = generate_partial_sliding_window(window_size=(S - 226) // window_size_factor, vtok_len=S - 226)
                    # block_mask = create_block_mask(sliding_window_mask, None, None, S, S, device="cuda", _compile=True, BLOCK_SIZE=(S - 226) // (window_size_factor * 2))
                    block_mask = create_block_mask(sliding_window_mask, None, None, S, S, device="cuda", _compile=True, BLOCK_SIZE=128)
                    self.block_mask[window_size_factor] = block_mask
                hidden_states = self.window_func(
                    query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2), block_mask=self.block_mask[window_size_factor]
                ).transpose(1, 2)
                torch.cuda.synchronize()
                # with residual share (latency *1.5)
                if "without" not in candidate:
                    hidden_states = hidden_states + self.cached_residual[window_size_factor].to(hidden_states.device)
            # output share
            elif candidate == 'output_share':
                hidden_states = self.prev_calib_output

            # add cfg sharing
            if "cfg_share" in candidate:
                hidden_states = hidden_states[:B//2].repeat(2, 1, 1, 1)

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
            wt = torch.ones(48, dtype=torch.int64) * ((S - 226) * 2)
            self.output_share_dict[self.stepi] = []
            for head_method in head_method_list:
                head = head_method[0]
                ws = head_method[1]
                if ws == "0":
                    wt[head] = 0
                    self.output_share_dict[self.stepi].append(head)
                else:
                    wt[head] = (S - 226) // int(ws)
            self.wt[self.stepi] = wt

            if self.stepi not in self.timestep_block_mask.keys():
                print(f"create block mask for {self.stepi}")
                sliding_window_per_head_mask = generate_partial_sliding_window_per_head(wt.to(query.device), vtok_len=S - 226)
                block_mask = create_block_mask(sliding_window_per_head_mask, None, H, S, S, device="cuda", _compile=True, BLOCK_SIZE=128)
            # if attn.name == "transformer_blocks.7.attn":
            #     breakpoint()
                self.timestep_block_mask[self.stepi] = block_mask
            output = self.window_func(
                query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2), block_mask=self.timestep_block_mask[self.stepi]
            ).transpose(1, 2)
            torch.cuda.synchronize()

        else:
            self.asc_enable = True
            wt = torch.ones(48, dtype=torch.int64) * ((S - 226) * 2)
            cfg_mask = [False] * 48
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
                    wt[head] = (S - 226) // abs(int(ws))
            self.wt[self.stepi] = wt
            if self.cfg_mask is None:
                self.cfg_mask = {}
            self.cfg_mask[self.stepi] = cfg_mask
            if self.stepi not in self.timestep_block_mask.keys():
                print(f"create block mask for {self.stepi}")
                if any(cfg_mask):
                    print("asc adopted")
                    sliding_window_per_head_mask_asc = generate_partial_sliding_window_per_head(wt.to(query.device)[cfg_mask], vtok_len=S - 226)
                    block_mask_asc = create_block_mask(sliding_window_per_head_mask_asc, None, cfg_mask.sum().item(), S, S, device="cuda", _compile=True, BLOCK_SIZE=128)
                else:
                    print("asc not adopted")
                    block_mask_asc = None
                sliding_window_per_head_mask = generate_partial_sliding_window_per_head(wt.to(query.device)[~cfg_mask], vtok_len=S - 226)
                block_mask = create_block_mask(sliding_window_per_head_mask, None, H - cfg_mask.sum().item(), S, S, device="cuda", _compile=True, BLOCK_SIZE=128)
                self.timestep_block_mask[self.stepi] = (block_mask, block_mask_asc)
            
            if block_mask_asc is not None:
                output = headwise_cfg_attention(query, key, value, self.window_func, cfg_mask, self.timestep_block_mask[self.stepi])
            else:
                output = self.window_func(
                    query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2), block_mask=self.timestep_block_mask[self.stepi][0]
                ).transpose(1, 2)
        # breakpoint()

        # recalculate attention
        # output, residual = full_mm_attn_with_window_residual(query.transpose(1,2), key.transpose(1,2), value.transpose(1,2), 
        #                                            vis_token_len=S-226, 
        #                                            txt_token_len=226, 
        #                                            left_window_size=wt.cuda(),
        #                                            right_window_size=wt.cuda(),
        #                                            block_size=128)
        # output = output.transpose(1,2)
        # if self.stepi not in self.timestep_block_mask.keys():
        #     print(f"create block mask for {self.stepi}")
        #     sliding_window_per_head_mask = generate_partial_sliding_window_per_head(wt.to(query.device), vtok_len=S - 226)
        #     block_mask = create_block_mask(sliding_window_per_head_mask, None, H, S, S, device="cuda", _compile=True, BLOCK_SIZE=128)
        #     # if attn.name == "transformer_blocks.7.attn":
        #     #     breakpoint()
        #     self.timestep_block_mask[self.stepi] = block_mask
        # output = self.window_func(
        #     query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2), block_mask=self.timestep_block_mask[self.stepi]
        # ).transpose(1, 2)
        # torch.cuda.synchronize()
        # B S H D
        output[:,:,self.output_share_dict[self.stepi],:] = self.prev_calib_output[:,:,self.output_share_dict[self.stepi],:]
        return output
        
        
    def get_attn_weight_qkv_process_func(self, query, key, value, forward_args):
        # breakpoint()
        score=torch.matmul(query, key.transpose(-1, -2))
        p=F.softmax(score, dim=-1) # b,head,seq,seq
        self.attn_weight+=p.mean(0).detach()
        self.attn_weight_num_count+=1
        
        hidden_states = flash_attn.flash_attn_func(query, key, value)
        return hidden_states
    
    def raw_qkv_process_func(self, query, key, value, forward_args):
        hidden_states = flash_attn.flash_attn_func(query, key, value)
        return hidden_states
    
    def raw_residual_cache_qkv_process_func(self, query, key, value, forward_args):
        hidden_states = flash_attn.flash_attn_func(query, key, value)
        attn=forward_args["attn"]
        _, S, _, _ = query.shape
        if self.block_mask is None:
            self.block_mask = {}
        candidates = self.dfa_config.get_available_candidates(attn.name)
        for candidate in candidates:
            if "window_attn" in candidate and "without_residual" not in candidate:
                self.curr_window_factor = int(candidate.split("_")[-1])
                if self.curr_window_factor not in self.block_mask.keys():
                    print(f"Compile window attention kernel 1")
                    window_size = (S - 226) // self.curr_window_factor
                    sliding_window_mask = generate_partial_sliding_window(window_size=window_size, vtok_len=S - 226)
                    # block_mask = create_block_mask(sliding_window_mask, None, None, S, S, device="cuda", _compile=True, BLOCK_SIZE=window_size//2)
                    block_mask = create_block_mask(sliding_window_mask, None, None, S, S, device="cuda", _compile=True, BLOCK_SIZE=128)
                    self.block_mask[self.curr_window_factor] = block_mask
                w_hidden_states = self.window_func(
                    query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2), block_mask=self.block_mask[self.curr_window_factor]
                ).transpose(1, 2)
                torch.cuda.synchronize()
                self.cached_residual[self.curr_window_factor] = (hidden_states - w_hidden_states).detach().cpu()
        return hidden_states
    
    def window_attn_qkv_process_func(self, query, key, value, forward_args):
        _, S, _, _ = query.shape
        if self.block_mask is None:
            self.block_mask = {}
        if self.curr_window_factor not in self.block_mask.keys():
            print(f"Compile window attention kernel 2")
            window_size = (S - 226) // self.curr_window_factor
            sliding_window_mask = generate_partial_sliding_window(window_size=window_size, vtok_len=S - 226)
            # block_mask = create_block_mask(sliding_window_mask, None, None, S, S, device="cuda", _compile=True, BLOCK_SIZE=window_size // 2)
            block_mask = create_block_mask(sliding_window_mask, None, None, S, S, device="cuda", _compile=True, BLOCK_SIZE=128)
            self.block_mask[self.curr_window_factor] = block_mask
        hidden_states = self.window_func(
            query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2), block_mask=self.block_mask[self.curr_window_factor]
        ).transpose(1, 2)
        torch.cuda.synchronize()
        hidden_states = hidden_states + self.cached_residual[self.curr_window_factor].to(hidden_states.device)
        # Add residual
        return hidden_states
    
    def window_attn_no_residual_qkv_process_func(self, query, key, value, forward_args):
        _, S, _, _ = query.shape
        if self.block_mask is None:
            self.block_mask = {}
        if self.curr_window_factor not in self.block_mask.keys():
            print(f"Compile window attention kernel 2")
            window_size = (S - 226) // self.curr_window_factor
            sliding_window_mask = generate_partial_sliding_window(window_size=window_size, vtok_len=S - 226)
            # block_mask = create_block_mask(sliding_window_mask, None, None, S, S, device="cuda", _compile=True, BLOCK_SIZE=window_size//2)
            block_mask = create_block_mask(sliding_window_mask, None, None, S, S, device="cuda", _compile=True, BLOCK_SIZE=128)
            self.block_mask[self.curr_window_factor] = block_mask
        hidden_states = self.window_func(
            query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2), block_mask=self.block_mask[self.curr_window_factor]
        ).transpose(1, 2)
        torch.cuda.synchronize()
        # Add residual
        return hidden_states
    
    def qkv_process_perhead_func(self, query, key, value, forward_args):
        # breakpoint()
        _, S, _, _ = query.shape
        # wt = self.wt[self.stepi]
        # output, residual = full_mm_attn_with_window_residual(query.transpose(1,2), key.transpose(1,2), value.transpose(1,2), 
        #                                            vis_token_len=S-226, 
        #                                            txt_token_len=226, 
        #                                            left_window_size=wt.cuda(),
        #                                            right_window_size=wt.cuda(),
        #                                            block_size=128)
        # output = output.transpose(1,2)
        # residual = residual.transpose(1,2)
        # self.cached_output = output
        # self.cached_residual = residual
        # breakpoint()
        if self.asc_enable:
            if self.timestep_block_mask[self.stepi][1] is not None:
                output = headwise_cfg_attention(query, key, value, self.window_func, self.cfg_mask[self.stepi], self.timestep_block_mask[self.stepi])
            else:
                output = self.window_func(
                    query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2), block_mask=self.timestep_block_mask[self.stepi][0]
                ).transpose(1, 2)
        else:
            output = self.window_func(
                query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2), block_mask=self.timestep_block_mask[self.stepi]
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
        hidden_states = flash_attn.flash_attn_func(query, key, value)
        if self.stepi+1 in self.output_share_dict.keys():
            self.cached_output = hidden_states[:,:,self.output_share_dict[self.stepi+1],:]
        return hidden_states
            
    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.FloatTensor:
        if self.forward_mode == "calib_collect_info":
            if self.stepi != 0:
            # qkv_process_func=self.calib_collect_info_qkv_process_func
                qkv_process_func=  self.calib_qkv_process_func
            else:
                qkv_process_func=self.raw_qkv_process_func
        
        elif self.forward_mode == "normal":
            if self.steps_method[self.stepi] == "raw":
                if self.raw_steps_residual_config[self.stepi][0] == True:
                    qkv_process_func = self.raw_residual_cache_qkv_process_func
                else:
                    qkv_process_func=self.raw_qkv_process_func
            elif self.steps_method[self.stepi] == "output_share":
                return self.cached_output
            elif "window_attn" in self.steps_method[self.stepi]:
                self.curr_window_factor = int(self.steps_method[self.stepi].split("_")[-1])
                if "without" not in self.steps_method[self.stepi]:
                    qkv_process_func=self.window_attn_qkv_process_func
                else:
                    qkv_process_func=self.window_attn_no_residual_qkv_process_func

        elif self.forward_mode == "perhead_normal":
            # print(self.stepi)
            if self.stepi == 0:
                # qkv_process_func = self.raw_qkv_process_after_calib_func
                qkv_process_func = self.raw_qkv_process_after_calib_func
            else:
                # qkv_process_func = self.qkv_process_perhead_func
                qkv_process_func = self.qkv_process_perhead_func
        
        hidden_states, encoder_hidden_states = self.run_forward(attn, hidden_states, encoder_hidden_states, qkv_process_func, image_rotary_emb)
        self.curr_window_factor = None
        return hidden_states, encoder_hidden_states
        
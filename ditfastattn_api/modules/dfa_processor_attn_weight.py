import torch
from diffusers.models.attention_processor import Attention, AttnProcessor2_0, JointAttnProcessor2_0
from typing import Optional
import torch.nn.functional as F
import flash_attn
import copy
from time import time

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


class MMDiTFastAttnProcessor:
    def __init__(self, steps_method=None, cond_first=None, window_func=None):
        self.steps_method = steps_method
        self.cond_first = cond_first
        self.window_func = window_func
        self.forward_mode = "normal" # could be "normal" or "calib_get_grad" or "calib_post_inference"
        self.raw_steps_residual_config = self.compute_raw_steps_residual_config(steps_method)
        self.block_mask = None
        self.compression_influences = {}
        self.prev_calib_output = None
        self.cached_residual = None
        self.cached_output = None
        self.dfa_config=None
        self.stepi=0
        
        # self.attn_weight=0
        # self.attn_weight_num_count=0
        window_size_candidates = [0.5,0.4,0.3,0.2,0.1]
        self.relative_MSE_threshold=0.01 # hyperparameter
        self.evaluated_latency=None

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
        self, query: torch.Tensor, key: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, scale
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
    
    def run_forward(self, attn:Attention, hidden_states, encoder_hidden_states, qkv_process_func):
        
        forward_args = dict(attn=attn, hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states)
        
        residual = hidden_states
        
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        context_input_ndim = encoder_hidden_states.ndim
        if context_input_ndim == 4:
            batch_size, channel, height, width = encoder_hidden_states.shape
            encoder_hidden_states = encoder_hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size = encoder_hidden_states.shape[0]

        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)
        
        encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
        encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
        encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)
        
        query = torch.cat([query, encoder_hidden_states_query_proj], dim=1)
        key = torch.cat([key, encoder_hidden_states_key_proj], dim=1)
        value = torch.cat([value, encoder_hidden_states_value_proj], dim=1)
        
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim)
        key = key.view(batch_size, -1, attn.heads, head_dim)
        value = value.view(batch_size, -1, attn.heads, head_dim)
        
        # attn
        
        hidden_states = qkv_process_func(query, key, value, forward_args)
        if self.forward_mode == "calib_post_inference":
            self.prev_calib_output = hidden_states.detach()
        
        hidden_states = hidden_states.reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)
        hidden_states, encoder_hidden_states = (
            hidden_states[:, : residual.shape[1]],
            hidden_states[:, residual.shape[1] :],
        )
        
        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)
        
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        
        if not attn.context_pre_only:
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)
        
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        if context_input_ndim == 4:
            encoder_hidden_states = encoder_hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )

        if self.forward_mode == "normal" or self.forward_mode == "calib_post_inference":
            self.cached_output = hidden_states.detach(), encoder_hidden_states.detach()
        
        return hidden_states, encoder_hidden_states
    
    def calib_collect_info_qkv_process_func(self, query, key, value, forward_args):
        # witout residual share window attention
        pass
    
    def calib_qkv_process_func(self, query, key, value, forward_args):
        # window_size_candidates = [0.995,0.99,0.98,0.97,0.96,0.95]
        headwise_relative_MSE={} # key (head, method) value: relative MSE = 均方误差（MSE）与目标变量方差的比值 （maybe）
        latency_delta={}
        attn=forward_args["attn"]
        candidates = self.dfa_config.get_available_candidates(attn.name)
        _, S, _, _ = query.shape
        full_hidden_states = flash_attn.flash_attn_func(query, key, value)
        for candidate in candidates:
            if "window_attn" in candidate:
                window_size_factor = candidate.split("_")[-1]
                if window_size_factor not in self.block_mask.keys():
                    sliding_window_mask = generate_partial_sliding_window(window_size=(S - 333) // window_size_factor, vtok_len=S - 333)
                    block_mask = create_block_mask(sliding_window_mask, None, None, S, S, device="cuda", _compile=True, BLOCK_SIZE=128)
                    self.block_mask[window_size_factor] = block_mask
                hidden_states = self.window_func(
                    query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2), block_mask=self.block_mask[window_size_factor]
                ).transpose(1, 2)
                torch.cuda.synchronize()
                if "without" not in candidate:
                    hidden_states = hidden_states + self.cached_residual
            elif candidate == 'output_share':
                hidden_states = self.prev_calib_output
            breakpoint()
            rse = ((hidden_states - full_hidden_states)**2).mean(keepdim=-1) / ((full_hidden_states - full_hidden_states.mean(keepdim=-1))**2).mean()
            self.headwise_relative_MSE[candidate] = rse


        # without residual share
        
        # with residual share (latency *1.5)
        
        # output share
        
        # ILP: target is to minimize the latency given MSE is below a threshold (?)
        
        # per head, gen kernel

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
        _, S, _, _ = query.shape
        if self.block_mask is None:
            print(f"Compile window attention kernel 1")
            sliding_window_mask = generate_partial_sliding_window(window_size=(S - 333) // 8, vtok_len=S - 333)
            block_mask = create_block_mask(sliding_window_mask, None, None, S, S, device="cuda", _compile=True, BLOCK_SIZE=128)
            self.block_mask = block_mask
        w_hidden_states = self.window_func(
            query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2), block_mask=self.block_mask
        ).transpose(1, 2)
        torch.cuda.synchronize()
        self.cached_residual = (hidden_states - w_hidden_states).detach()
        return hidden_states
    
    def window_attn_qkv_process_func(self, query, key, value, forward_args):
        _, S, _, _ = query.shape
        if self.block_mask is None:
            print(f"Compile window attention kernel 2")
            sliding_window_mask = generate_partial_sliding_window(window_size=(S - 333) // 8, vtok_len=S - 333)
            block_mask = create_block_mask(sliding_window_mask, None, None, S, S, device="cuda", _compile=True, BLOCK_SIZE=128)
            self.block_mask = block_mask
        hidden_states = self.window_func(
            query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2), block_mask=self.block_mask
        ).transpose(1, 2)
        torch.cuda.synchronize()
        hidden_states = hidden_states + self.cached_residual
        # Add residual
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
        if self.forward_mode == "calib_collect_info":
            qkv_process_func=self.calib_collect_info_qkv_process_func
            
        elif self.forward_mode == "calib_post_inference":
            # breakpoint()
            qkv_process_func=  self.calib_qkv_process_func
            # if self.steps_method[self.stepi] == "raw":
            #     qkv_process_func=  self.raw_residual_cache_qkv_process_func
            # elif self.steps_method[self.stepi] == "output_share":
            #     return self.cached_output
            # elif "residual_window_attn" in self.steps_method[self.stepi]:
            #     qkv_process_func=self.window_attn_qkv_process_func
        
        elif self.forward_mode == "normal":
            if self.steps_method[self.stepi] == "raw":
                if self.raw_steps_residual_config[self.stepi][0] == True:
                    qkv_process_func = self.raw_residual_cache_qkv_process_func
                else:
                    qkv_process_func=self.raw_qkv_process_func
            elif self.steps_method[self.stepi] == "output_share":
                return self.cached_output
            elif "residual_window_attn" in self.steps_method[self.stepi]:
                qkv_process_func=self.window_attn_qkv_process_func
        
        hidden_states, encoder_hidden_states = self.run_forward(attn, hidden_states, encoder_hidden_states, qkv_process_func)
        return hidden_states, encoder_hidden_states
        
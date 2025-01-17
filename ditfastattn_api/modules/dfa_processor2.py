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


class MMDiTFastAttnProcessor:
    def __init__(self, steps_method=None, cond_first=None, window_func=None):
        self.steps_method = steps_method
        self.cond_first = cond_first
        self.window_func = window_func
        self.forward_mode = "normal" # could be "normal" or "get_grad"
        
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
        return hidden_states, encoder_hidden_states

    def get_grad_qkv_process_func(self, query, key, value, forward_args):
        hidden_states = flash_attn.flash_attn_func(query, key, value)
        attn=forward_args["attn"]
        
        def get_grad_hook(grad):
            with torch.no_grad():
                candidates = self.dfa_config.get_available_candidates(attn.name)
                for candidate in candidates:
                    if candidate == 'output_share':
                        influence = ((self.prev_calib_output.float() - hidden_states.float()) * grad[0].detach()).sum().abs().cpu().numpy()
                    elif candidate == 'window_attn':
                        hidden_states_with_wars= self.window_attn_qkv_process_func(query, key, value, attn, forward_args)
                        influence = ((hidden_states_with_wars.float() - hidden_states.float()) * grad[0].detach()).sum().abs().cpu().numpy()
                    if candidate not in attn.compression_influences:
                        attn.compression_influences[candidate] = 0
                    attn.compression_influences[candidate] += influence
        hidden_states.register_backward_hook(get_grad_hook)
        return hidden_states
    
    def qkv_process_save_hidden_state_for_next_step_calib(self, query, key, value, forward_args, func):
        hidden_states = func(query, key, value)
        self.xxxx=hidden_states
        return hidden_states
    
    def raw_qkv_process_func(self, query, key, value, forward_args):
        hidden_states = flash_attn.flash_attn_func(query, key, value)
        return hidden_states
    
    def window_attn_qkv_process_func(self, query, key, value, forward_args):
        hidden_states = flash_attn.flash_attn_func(query, key, value)
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
        if self.forward_mode == "calib_get_grad":
            qkv_process_func=self.get_grad_qkv_process_func
            
        elif self.forward_mode == "calib_post_inference":
            if self.steps_method[self.stepi] == "raw":
                qkv_process_func=self.raw_qkv_process_func
            elif self.steps_method[self.stepi] == "output_share":
                pass
            elif "residual_window_attn" in self.steps_method[self.stepi]:
                qkv_process_func=self.window_attn_qkv_process_func
                
        elif self.forward_mode == "normal":
            if self.steps_method[self.stepi] == "raw":
                qkv_process_func=self.raw_qkv_process_func
            elif self.steps_method[self.stepi] == "output_share":
                pass
            elif "residual_window_attn" in self.steps_method[self.stepi]:
                qkv_process_func=self.window_attn_qkv_process_func
        
        self.run_forward(attn, hidden_states, encoder_hidden_states,qkv_process_func)
        
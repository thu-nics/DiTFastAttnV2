import torch
from diffusers.models.attention_processor import Attention, AttnProcessor2_0, JointAttnProcessor2_0
from tmp.full_window_attn_with_window_residual import full_mm_attn_with_window_residual
from typing import List, Optional
import torch.nn.functional as F
import flash_attn_ours
import copy
from time import time
from ditfastattn_api.modules.ilp import solve_ip

# from natten.functional import na1d, na2d
import torch.nn as nn

from torch.nn.attention.flex_attention import _mask_mod_signature, or_masks
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
from functools import partial

class FLUXFastAttnProcessor:
    def __init__(self, steps_method=None, window_func=None, alpha=0):
        self.steps_method = steps_method
        self.window_func = window_func
        self.forward_mode = "normal" # could be "normal" or "calib_get_grad" or "calib_post_inference"
        self.block_mask = {}
        self.prev_calib_output = None
        self.cached_residual = {}
        self.cached_output = None
        self.dfa_config=None
        self.stepi=0
        self.curr_window_factor = None
        self.alpha = alpha
        self.wt = {}
        self.timestep_block_mask = {}
        self.output_share_dict = {}
        
        
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
        # breakpoint()
        # shortbreak - layer output caching

        
        forward_args = dict(attn=attn, hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states)
        
        batch_size, _, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        head_dim = hidden_states.shape[-1] // attn.heads

        if self.stepi in self.output_share_dict:
            if len(self.output_share_dict[self.stepi]) == 24:
                hidden_states = self.cached_output
                if self.stepi+1 in self.output_share_dict.keys():
                    self.cached_output = hidden_states[:,:,self.output_share_dict[self.stepi+1],:].detach()
                hidden_states = hidden_states.reshape(batch_size, -1, attn.heads * head_dim)
                if encoder_hidden_states is not None:
                    encoder_hidden_states, hidden_states = (
                        hidden_states[:, : encoder_hidden_states.shape[1]],
                        hidden_states[:, encoder_hidden_states.shape[1] :],
                    )

                    # linear proj
                    hidden_states = attn.to_out[0](hidden_states)
                    # dropout
                    hidden_states = attn.to_out[1](hidden_states)

                    encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

                    return hidden_states, encoder_hidden_states
                else:
                    return hidden_states

        # `sample` projections.
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # the attention in FluxSingleTransformerBlock does not use `encoder_hidden_states`
        if encoder_hidden_states is not None:
            # `context` projections.
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

            # attention
            query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
            key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
            value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)

        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb

            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)
        
        hidden_states = qkv_process_func(query.transpose(1,2), key.transpose(1,2), value.transpose(1,2), forward_args)
        if self.forward_mode == "calib_collect_info":
            self.prev_calib_output = hidden_states.detach().cpu()
        
        hidden_states = hidden_states.reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = (
                hidden_states[:, : encoder_hidden_states.shape[1]],
                hidden_states[:, encoder_hidden_states.shape[1] :],
            )

            # linear proj
            hidden_states = attn.to_out[0](hidden_states)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)

            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

            return hidden_states, encoder_hidden_states
        else:
            return hidden_states
    
    def calib_collect_info_qkv_process_func(self, query, key, value, forward_args):
        # witout residual share window attention
        pass
    
    def calib_qkv_process_func(self, query, key, value, forward_args):
        # window_size_candidates = [0.995,0.99,0.98,0.97,0.96,0.95]
        headwise_relative_MSE={} # key (head, method) value: relative MSE = 均方误差（MSE）与目标变量方差的比值 （maybe）
        # latency_delta={}
        attn=forward_args["attn"]
        candidates = self.dfa_config.get_available_candidates(attn.name)
        _, S, H, _ = query.shape
        # full_hidden_states = flash_attn.flash_attn_func(query, key, value)
        full_hidden_states = F.scaled_dot_product_attention(
            query.transpose(1,2), key.transpose(1,2), value.transpose(1,2), attn_mask=None, dropout_p=0.0, is_causal=False
        ).transpose(1,2)
        for candidate in candidates:
            if "window_attn" in candidate:
                # without residual share
                window_size_factor = int(candidate.split("_")[-1])
                window_size = (S - 512) // (window_size_factor * 2)
                hidden_states = flash_attn_ours.headwise_arrow_attn_trans(
                    query, 
                    key, 
                    value, 
                    window_sizes = torch.ones((H, 2), device=query.device, dtype=torch.int32) * window_size, 
                    seqlen_q_vision = S - 512, 
                    seqlen_k_vision = S - 512
                )
                torch.cuda.synchronize()
                # with residual share (latency *1.5)
                if "without" not in candidate:
                    hidden_states = hidden_states + self.cached_residual[window_size_factor].to(hidden_states.device)
            # output share
            elif candidate == 'output_share':
                hidden_states = self.prev_calib_output.to(query.device)
            # print(((full_hidden_states - full_hidden_states.mean(dim=(0,1,3), keepdim=True))**2))
            rse = (((hidden_states - full_hidden_states)**2).mean(dim=(0,1,3)) / ((full_hidden_states - full_hidden_states.mean(dim=(0,1,3), keepdim=True))**2).mean(dim=(0,1,3))).detach().cpu()
            # print(rse)
            for h in range(rse.size()[0]):
                if rse[h].item() < self.alpha / H * 1.5:
                    headwise_relative_MSE[(h, candidate)] = rse[h].item()
            
        # ILP: target is to minimize the latency given MSE is below a threshold (?)
        # breakpoint()
        if self.stepi not in self.dfa_config.plan.keys():
            self.dfa_config.plan[self.stepi] = {}
        if bool(headwise_relative_MSE):
            # ILP: target is to minimize the latency given MSE is below a threshold (?)
            head_method_list = solve_ip(headwise_relative_MSE,self.dfa_config.latency, self.alpha)
            print(head_method_list)
        else:
            head_method_list = {}
        self.dfa_config.plan[self.stepi][attn.name] = head_method_list

        # breakpoint()
        # per head, gen kernel
        wt = -torch.ones(24, dtype=torch.int32, device=query.device)
        # wt = torch.ones(24, dtype=torch.int64) * ((S - 512) * 2)
        self.output_share_dict[self.stepi] = []
        for head_method in head_method_list:
            head = head_method[0]
            ws = head_method[1]
            if ws == "0":
                wt[head] = 0
                self.output_share_dict[self.stepi].append(head)
            else:
                wt[head] = (S - 512) // (int(ws)*2)
        wt = wt.repeat_interleave(2, dim=0).view(-1, 2)
        # breakpoint()

        self.wt[self.stepi] = wt

        output = flash_attn_ours.headwise_arrow_attn_trans(
            query, 
            key, 
            value, 
            window_sizes=wt,
            seqlen_q_vision = S - 512,
            seqlen_k_vision = S - 512
        )
        # B S H D
        output[:,:,self.output_share_dict[self.stepi],:] = self.prev_calib_output.to(query.device)[:,:,self.output_share_dict[self.stepi],:]
        return output

    
    def raw_qkv_process_func(self, query, key, value, forward_args):
        hidden_states = flash_attn_ours.flash_attn_func(query, key, value)
        return hidden_states

    # def raw_qkv_process_func(self, query, key, value, forward_args):
    #     hidden_states = F.scaled_dot_product_attention(
    #         query.transpose(1,2), key.transpose(1,2), value.transpose(1,2), attn_mask=None, dropout_p=0.0, is_causal=False
    #     ).transpose(1,2)
    #     return hidden_states
    
    def qkv_process_perhead_func(self, query, key, value, forward_args):
        _, S, _, _ = query.shape
        # output = self.window_func(
        #     query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2), block_mask=self.timestep_block_mask[self.stepi]
        # ).transpose(1, 2)
        # torch.cuda.synchronize()
        output = flash_attn_ours.headwise_arrow_attn_trans(
            query,
            key,
            value,
            window_sizes = self.wt[self.stepi],
            seqlen_q_vision = S - 512,
            seqlen_k_vision = S - 512
        )
        
        if output[:,:,self.output_share_dict[self.stepi],:].shape != self.cached_output.shape:
            breakpoint()
        output[:,:,self.output_share_dict[self.stepi],:] = self.cached_output
        if self.stepi+1 in self.output_share_dict.keys():
            self.cached_output = output[:,:,self.output_share_dict[self.stepi+1],:].detach()
        return output
    
    def raw_qkv_process_after_calib_func(self, query, key, value, forward_args):
        # hidden_states = F.scaled_dot_product_attention(
        #     query.transpose(1,2), key.transpose(1,2), value.transpose(1,2), attn_mask=None, dropout_p=0.0, is_causal=False
        # ).transpose(1,2)
        hidden_states = flash_attn_ours.flash_attn_func(query, key, value)
        if self.stepi+1 in self.output_share_dict.keys():
            self.cached_output = hidden_states[:,:,self.output_share_dict[self.stepi+1],:].detach()
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
            qkv_process_func=self.raw_qkv_process_func

        elif self.forward_mode == "perhead_normal":
            # print(self.stepi)
            if self.stepi == 0:
                qkv_process_func = self.raw_qkv_process_after_calib_func
            else:
                qkv_process_func = self.qkv_process_perhead_func
        if encoder_hidden_states is not None:
            hidden_states, encoder_hidden_states = self.run_forward(attn, hidden_states, encoder_hidden_states, qkv_process_func, image_rotary_emb)
            self.curr_window_factor = None
            return hidden_states, encoder_hidden_states
        else:
            hidden_states = self.run_forward(attn, hidden_states, encoder_hidden_states, qkv_process_func, image_rotary_emb)
            self.curr_window_factor = None
            return hidden_states
        

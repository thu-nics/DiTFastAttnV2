import torch
from diffusers.models.attention_processor import Attention, AttnProcessor2_0
from typing import Optional
import torch.nn.functional as F
import flash_attn
import copy

# from natten.functional import na1d, na2d
import torch.nn as nn

from torch.nn.attention.flex_attention import _mask_mod_signature, or_masks
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
from functools import partial


# flex attn partial window attention
def generate_partial_sliding_window(window_size: int, vtok_len: int) -> _mask_mod_signature:
    """Generates a sliding window attention mask with a given window size.
    Args:
        window_size: The size of the sliding window.

    Note:
        We assume that the window size represents the lookback size and we mask out all future tokens
        similar to causal masking.
    """

    def sliding_window_with_offset(b, h, q_idx, kv_idx):
        return torch.abs(q_idx - kv_idx) <= window_size // 2

    def partial_full(b, h, q_idx, kv_idx):
        return (q_idx >= vtok_len) | (kv_idx >= vtok_len)

    sliding_window_mask = or_masks(partial_full, sliding_window_with_offset)
    sliding_window_mask.__name__ = f"sliding_window_{window_size}"
    return sliding_window_mask


class DiTFastAttnProcessor:
    def __init__(self, steps_method=None, cond_first=None):
        self.steps_method = steps_method
        # CFG order flag (conditional first or unconditional first)
        self.cond_first = cond_first
        # Check at which timesteps do we need to compute the full-window residual
        # of this attention module
        self.raw_steps_residual_config = self.compute_raw_steps_residual_config(steps_method)
        self.need_cache_output = True
        self.mask_cache = {}
        self.stepi = 0

    def update_config(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, copy.deepcopy(value))
        self.raw_steps_residual_config = self.compute_raw_steps_residual_config(self.steps_method)

    def compute_raw_steps_residual_config(self, steps_method):
        steps_residual_config = []
        assert steps_method[0] == "raw", "The first step of DiTFastAttnProcessor must be raw"
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

    def run_forward_method(self, m, hidden_states, encoder_hidden_states, attention_mask, temb, method):
        residual = hidden_states
        if method == "output_share":
            hidden_states = m.cached_output
        else:
            if "cfg_attn_share" in method:
                # Directly use the unconditional branch's attention output
                # as the conditional branch's attention output

                # TODO: Maybe use the conditional branch's attention output
                # as the unconditional's is better
                batch_size = hidden_states.shape[0]
                if self.cond_first:
                    hidden_states = hidden_states[: batch_size // 2]
                else:
                    hidden_states = hidden_states[batch_size // 2 :]
                if encoder_hidden_states is not None:
                    if self.cond_first:
                        encoder_hidden_states = encoder_hidden_states[: batch_size // 2]
                    else:
                        encoder_hidden_states = encoder_hidden_states[batch_size // 2 :]
                if attention_mask is not None:
                    if self.cond_first:
                        attention_mask = attention_mask[: batch_size // 2]
                    else:
                        attention_mask = attention_mask[batch_size // 2 :]

            if m.spatial_norm is not None:
                hidden_states = m.spatial_norm(hidden_states, temb)

            input_ndim = hidden_states.ndim

            if input_ndim == 4:
                batch_size, channel, height, width = hidden_states.shape
                hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

            batch_size, sequence_length, _ = (
                hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
            )
            if attention_mask is not None:
                attention_mask = m.prepare_attention_mask(attention_mask, sequence_length, batch_size)
                # scaled_dot_product_attention expects attention_mask shape to be
                # (batch, heads, source_length, target_length)
                attention_mask = attention_mask.view(batch_size, m.heads, -1, attention_mask.shape[-1])

            if m.group_norm is not None:
                hidden_states = m.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

            query = m.to_q(hidden_states)

            if encoder_hidden_states is None:
                encoder_hidden_states = hidden_states
            elif m.norm_cross:
                encoder_hidden_states = m.norm_encoder_hidden_states(encoder_hidden_states)

            key = m.to_k(encoder_hidden_states)
            value = m.to_v(encoder_hidden_states)

            inner_dim = key.shape[-1]
            head_dim = inner_dim // m.heads

            query = query.view(batch_size, -1, m.heads, head_dim)

            key = key.view(batch_size, -1, m.heads, head_dim)
            value = value.view(batch_size, -1, m.heads, head_dim)

            if attention_mask is not None:
                assert "residual_window_attn" not in method

                hidden_states = F.scaled_dot_product_attention(
                    query.transpose(1, 2),
                    key.transpose(1, 2),
                    value.transpose(1, 2),
                    attn_mask=attention_mask,
                    dropout_p=0.0,
                    is_causal=False,
                ).transpose(1, 2)
            elif "raw" in method:
                all_hidden_states = flash_attn.flash_attn_func(query, key, value)
                if self.raw_steps_residual_config[self.stepi][0] == True:
                    # Compute the full-window attention residual
                    window_size = self.raw_steps_residual_config[self.stepi][1]
                    w_hidden_states = flash_attn.flash_attn_func(query, key, value, window_size=window_size)
                    w_residual = all_hidden_states - w_hidden_states
                    if "cfg_attn_share" in method:
                        w_residual = torch.cat([w_residual, w_residual], dim=0)
                    # Save the residual for usage in follow-up steps
                    m.cached_residual = w_residual
                hidden_states = all_hidden_states
            elif "residual_window_attn" in method:
                window_size = int(method.split("_")[-1])
                w_hidden_states = flash_attn.flash_attn_func(
                    query, key, value, window_size=(window_size // 2, window_size // 2)
                )

                if "without_residual" in method:
                    # For ablation study of `residual_window_attn+without_residual`
                    # v.s. `residual_window_attn`
                    hidden_states = w_hidden_states
                else:
                    hidden_states = w_hidden_states + m.cached_residual[:batch_size].view_as(w_hidden_states)

            hidden_states = hidden_states.reshape(batch_size, -1, m.heads * head_dim)
            hidden_states = hidden_states.to(query.dtype)

            # linear proj
            hidden_states = m.to_out[0](hidden_states)
            # dropout
            hidden_states = m.to_out[1](hidden_states)

            if input_ndim == 4:
                hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

            if "cfg_attn_share" in method:
                hidden_states = torch.cat([hidden_states, hidden_states], dim=0)

            if self.need_cache_output:
                m.cached_output = hidden_states

        if m.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / m.rescale_output_factor

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
            hidden_states = self.run_opensora_forward_method(
                attn,
                hidden_states,
                encoder_hidden_states,
                attention_mask,
                temb,
                self.steps_method[self.stepi],
            )
        else:
            hidden_states = self.run_forward_method(
                attn,
                hidden_states,
                encoder_hidden_states,
                attention_mask,
                temb,
                self.steps_method[self.stepi],
            )
        # After been call once, add the timestep index of this attention module by 1
        self.stepi += 1
        return hidden_states


class JointAttnProcessor2_0:
    """Attention processor used typically in processing the SD3-like self-attention projections."""

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        *args,
        **kwargs,
    ) -> torch.FloatTensor:
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

        # `sample` projections.
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        # `context` projections.
        encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
        encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
        encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

        # attention
        query = torch.cat([query, encoder_hidden_states_query_proj], dim=1)
        key = torch.cat([key, encoder_hidden_states_key_proj], dim=1)
        value = torch.cat([value, encoder_hidden_states_value_proj], dim=1)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # Split the attention outputs.
        hidden_states, encoder_hidden_states = (
            hidden_states[:, : residual.shape[1]],
            hidden_states[:, residual.shape[1] :],
        )

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)
        if not attn.context_pre_only:
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        if context_input_ndim == 4:
            encoder_hidden_states = encoder_hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        return hidden_states, encoder_hidden_states


flex_kernels = {}


class MMDiTFastAttnProcessor:
    def __init__(self, steps_method=None, cond_first=None):
        self.steps_method = steps_method
        # CFG order flag (conditional first or unconditional first)
        self.cond_first = cond_first
        # Check at which timesteps do we need to compute the full-window residual
        # of this attention module
        self.raw_steps_residual_config = self.compute_raw_steps_residual_config(steps_method)
        self.need_cache_output = True
        self.mask_cache = {}
        self.stepi = 0

        # flex attn kernel compile
        kernel_name = "window_640"
        if kernel_name in flex_kernels.keys():
            self.window_func, self.block_mask = flex_kernels[kernel_name]
        else:
            sliding_window_mask = generate_partial_sliding_window(window_size=640, vtok_len=4096)
            self.block_mask = create_block_mask(sliding_window_mask, 2, 16, 4096, 4096, device="cuda", _compile=True)
            self.window_func = torch.compile(partial(flex_attention, block_mask=self.block_mask))
            flex_kernels[kernel_name] = self.window_func, self.block_mask

    def update_config(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, copy.deepcopy(value))
        self.raw_steps_residual_config = self.compute_raw_steps_residual_config(self.steps_method)

    def compute_raw_steps_residual_config(self, steps_method):
        steps_residual_config = []
        assert steps_method[0] == "raw", "The first step of DiTFastAttnProcessor must be raw"
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

    def run_forward_method(self, m, hidden_states, encoder_hidden_states, temb, method, attention_mask=None):
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

        if method == "output_share":
            hidden_states, encoder_hidden_states = m.cached_output
        else:
            if "cfg_attn_share" in method:
                # Directly use the unconditional branch's attention output
                # as the conditional branch's attention output

                # TODO: Maybe use the conditional branch's attention output
                # as the unconditional's is better
                batch_size = hidden_states.shape[0]
                if self.cond_first:
                    hidden_states = hidden_states[: batch_size // 2]
                else:
                    hidden_states = hidden_states[batch_size // 2 :]
                if encoder_hidden_states is not None:
                    if self.cond_first:
                        encoder_hidden_states = encoder_hidden_states[: batch_size // 2]
                    else:
                        encoder_hidden_states = encoder_hidden_states[batch_size // 2 :]

            input_ndim = hidden_states.ndim

            if input_ndim == 4:
                batch_size, channel, height, width = hidden_states.shape
                hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
            context_input_ndim = encoder_hidden_states.ndim
            if context_input_ndim == 4:
                batch_size, channel, height, width = encoder_hidden_states.shape
                encoder_hidden_states = encoder_hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

            batch_size = encoder_hidden_states.shape[0]

            query = m.to_q(hidden_states)
            key = m.to_k(hidden_states)
            value = m.to_v(hidden_states)

            # `context` projections.
            encoder_hidden_states_query_proj = m.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = m.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = m.add_v_proj(encoder_hidden_states)

            # attention
            query = torch.cat([query, encoder_hidden_states_query_proj], dim=1)
            key = torch.cat([key, encoder_hidden_states_key_proj], dim=1)
            value = torch.cat([value, encoder_hidden_states_value_proj], dim=1)

            inner_dim = key.shape[-1]
            head_dim = inner_dim // m.heads

            query = query.view(batch_size, -1, m.heads, head_dim)
            key = key.view(batch_size, -1, m.heads, head_dim)
            value = value.view(batch_size, -1, m.heads, head_dim)

            if attention_mask is not None:
                assert "residual_window_attn" not in method

                hidden_states = F.scaled_dot_product_attention(
                    query.transpose(1, 2),
                    key.transpose(1, 2),
                    value.transpose(1, 2),
                    dropout_p=0.0,
                    is_causal=False,
                ).transpose(1, 2)
            elif "raw" in method:
                all_hidden_states = flash_attn.flash_attn_func(query, key, value)
                if self.raw_steps_residual_config[self.stepi][0] == True:
                    # Compute the full-window attention residual
                    # partial window attn using flex attn

                    w_hidden_states = self.window_func(
                        query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2), block_mask=self.block_mask
                    ).transpose(1, 2)

                    # window_size = self.raw_steps_residual_config[self.stepi][1]
                    # w_hidden_states = flash_attn.flash_attn_func(query, key, value, window_size=window_size)
                    w_residual = all_hidden_states - w_hidden_states
                    if "cfg_attn_share" in method:
                        w_residual = torch.cat([w_residual, w_residual], dim=0)
                    # Save the residual for usage in follow-up steps
                    m.cached_residual = w_residual
                hidden_states = all_hidden_states
            elif "residual_window_attn" in method:
                w_hidden_states = self.window_func(
                    query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2), block_mask=self.block_mask
                ).transpose(1, 2)
                # window_size = int(method.split("_")[-1])
                # w_hidden_states = flash_attn.flash_attn_func(query, key, value, window_size=window_size)

                if "without_residual" in method:
                    # For ablation study of `residual_window_attn+without_residual`
                    # v.s. `residual_window_attn`
                    hidden_states = w_hidden_states
                else:
                    hidden_states = w_hidden_states + m.cached_residual[:batch_size].view_as(w_hidden_states)

            hidden_states = hidden_states.reshape(batch_size, -1, m.heads * head_dim)
            hidden_states = hidden_states.to(query.dtype)

            # Split the attention outputs.
            hidden_states, encoder_hidden_states = (
                hidden_states[:, : residual.shape[1]],
                hidden_states[:, residual.shape[1] :],
            )

            # linear proj
            hidden_states = m.to_out[0](hidden_states)
            # dropout
            hidden_states = m.to_out[1](hidden_states)

            if input_ndim == 4:
                hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

            if "cfg_attn_share" in method:
                hidden_states = torch.cat([hidden_states, hidden_states], dim=0)
                encoder_hidden_states = torch.cat([encoder_hidden_states, encoder_hidden_states], dim=0)

            if not m.context_pre_only:
                encoder_hidden_states = m.to_add_out(encoder_hidden_states)

            if input_ndim == 4:
                hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
            if context_input_ndim == 4:
                encoder_hidden_states = encoder_hidden_states.transpose(-1, -2).reshape(
                    batch_size, channel, height, width
                )

            if self.need_cache_output:
                m.cached_output = hidden_states, encoder_hidden_states

        return hidden_states, encoder_hidden_states

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
        hidden_states = self.run_forward_method(
            attn,
            hidden_states,
            encoder_hidden_states,
            temb,
            self.steps_method[self.stepi],
        )
        # After been call once, add the timestep index of this attention module by 1
        self.stepi += 1
        return hidden_states

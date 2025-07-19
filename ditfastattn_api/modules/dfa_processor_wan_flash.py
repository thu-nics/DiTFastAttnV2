import torch
from diffusers.models.attention_processor import Attention
from typing import List, Optional
import torch.nn.functional as F
# import dfav2
from dfav2 import flash_attn_func, headwise_window_attn
from ditfastattn_api.modules.ilp import solve_ip
from ditfastattn_api.video_reorder import sparse_head_placement, hidden_states_placement


def headwise_cfg_attention(query, key, value, head_mask, window_sizes):
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

        attention_first_half = headwise_window_attn(query_first_half.transpose(1,2), 
                                                               key_first_half.transpose(1,2), 
                                                               value_first_half.transpose(1,2), 
                                                               window_sizes=window_sizes[head_mask,:]
                                                              )
        attention_second_half = attention_first_half.clone()
        attention_replace = torch.cat([attention_first_half, attention_second_half], dim=0)  # shape: (B, S, H_replace, D)
    else:
        attention_replace = torch.empty(B, S, 0, D, device=query.device)  # 如果没有需要替换的 head，返回空张量

    # 对于不需要替换的 head，正常计算所有 batch 的 attention
    if query_normal.size(2) > 0:  # 如果有不需要替换的 head
        attention_first_half = headwise_window_attn(query_first_half.transpose(1,2), 
                                                               key_first_half.transpose(1,2), 
                                                               value_first_half.transpose(1,2), 
                                                               window_sizes=window_sizes[~head_mask,:],
                                                              )
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
    def __init__(self, steps_method=None, alpha=0, window_func=None):
        self.window_func = window_func
        self.steps_method = steps_method
        self.forward_mode = "normal" # could be "normal" or "calib_get_grad" or "calib_post_inference"
        self.prev_calib_output = None
        self.cached_output = None
        self.dfa_config=None
        self.stepi=0
        self.alpha = alpha
        self.wt = {}
        self.timestep_block_mask = {}
        self.relative_MSE_threshold=0.01 # hyperparameter
        self.output_share_dict = {}
        self.cfg_mask = None
        self.asc_enable = False

        self.num_frames = None
        self.height = None
        self.width = None

        self.reorder_mask = {}

    
    def run_forward(self, 
                    attn:Attention, 
                    hidden_states:torch.Tensor, 
                    encoder_hidden_states:Optional[torch.Tensor] = None, 
                    attention_mask: Optional[torch.Tensor] = None,
                    rotary_emb: Optional[torch.Tensor] = None,
                    qkv_process_func: Optional[callable] = None):
        
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

            hidden_states = qkv_process_func(query, key, value, forward_args)
            hidden_states_img = hidden_states_img.transpose(1, 2).flatten(2, 3)
            hidden_states_img = hidden_states_img.type_as(query)

        hidden_states = qkv_process_func(query, key, value, forward_args)
        if self.forward_mode == "calib_collect_info":
            self.prev_calib_output = hidden_states

        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.type_as(query)

        if hidden_states_img is not None:
            hidden_states = hidden_states + hidden_states_img

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states
    
    
    def calib_qkv_process_func(self, query, key, value, forward_args):
        headwise_relative_MSE={}
        attn=forward_args["attn"]
        candidates = self.dfa_config.get_available_candidates(attn.name)
        B, H, S, _ = query.shape
        full_hidden_states = flash_attn_func(query.transpose(1,2), 
                                                    key.transpose(1,2), 
                                                    value.transpose(1,2))
        for candidate in candidates:
            if "arrow_attn" in candidate:
                window_size_factor = int(candidate.split("_")[-1])
                window_size = S // (window_size_factor * 2)
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
                    hidden_states = headwise_window_attn(query_out.transpose(1,2), 
                                                        key_out.transpose(1,2), 
                                                        value_out.transpose(1,2), 
                                                        window_sizes=torch.ones((H, 2), device=query.device, dtype=torch.int32) * window_size,).transpose(1,2)
                    hidden_states_placement(hidden_states, 
                                            output_hidden_states, 
                                            is_reorder, 
                                            0, 
                                            self.num_frames, 
                                            self.height * self.width)
                    hidden_states = output_hidden_states.transpose(1, 2)
                else:
                    hidden_states = headwise_window_attn(query.transpose(1,2), 
                                                                key.transpose(1,2), 
                                                                value.transpose(1,2), 
                                                                window_sizes=torch.ones((H, 2), device=query.device, dtype=torch.int32) * window_size,)
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
            wt = -torch.ones(H, dtype=torch.int32, device=query.device)
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
                    wt[head] = S // (int(ws)*2)
            wt = wt.repeat_interleave(2, dim=0).view(-1, 2)
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

            output = headwise_window_attn(
                query_out.transpose(1, 2), 
                key_out.transpose(1, 2), 
                value_out.transpose(1, 2), 
                window_sizes=wt,
                seqlen_q_vision = -1, 
                seqlen_k_vision = -1
            ).transpose(1,2)

            hidden_states_placement(output, 
                                output_hidden_states, 
                                self.reorder_mask[self.stepi].repeat(B, 1),
                                0, 
                                self.num_frames, 
                                self.height * self.width)
            output = output_hidden_states.transpose(1,2)

        else:
            self.asc_enable = True
            wt = -torch.ones(H, device=query.device, dtype=torch.int32)
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
                        reorder_mask[head] = 1
                    wt[head] = S // abs(int(ws)*2)
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
            
            if any(cfg_mask):
                output = headwise_cfg_attention(query, key, value, cfg_mask, self.timestep_block_mask[self.stepi]).transpose(1, 2)
            else:
                output = headwise_window_attn(
                    query.transpose(1, 2), 
                    key.transpose(1, 2), 
                    value.transpose(1, 2), 
                    window_sizes=wt,
                ).transpose(1, 2)

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
        hidden_states = flash_attn_func(q = query.transpose(1,2), k = key.transpose(1,2), v = value.transpose(1,2))
        return hidden_states
    
    def qkv_process_perhead_func(self, query, key, value, forward_args):
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
                output = headwise_cfg_attention(query_out, key_out, value_out, self.cfg_mask[self.stepi], self.wt[self.stepi])
            else:
                output = headwise_window_attn(
                    query_out.transpose(1,2), 
                    key_out.transpose(1,2), 
                    value_out.transpose(1,2), 
                    window_sizes = self.wt[self.stepi],
                ).transpose(1, 2)
        else:
            output = headwise_window_attn(
                    query_out.transpose(1,2), 
                    key_out.transpose(1,2), 
                    value_out.transpose(1,2), 
                    window_sizes = self.wt[self.stepi],
                ).transpose(1, 2)
            
        hidden_states_placement(output, 
                                output_hidden_states, 
                                self.reorder_mask[self.stepi].repeat(B, 1),
                                0, 
                                self.num_frames, 
                                self.height * self.width)
        output = output_hidden_states.transpose(1,2)

        output[:,:,self.output_share_dict[self.stepi],:] = self.cached_output
        if self.stepi+1 in self.output_share_dict.keys():
            self.cached_output = output[:,:,self.output_share_dict[self.stepi+1],:]
        return output
    
    
    def raw_qkv_process_after_calib_func(self, query, key, value, forward_args):
        hidden_states = flash_attn_func(query.transpose(1,2), 
                                                        key.transpose(1,2), 
                                                        value.transpose(1,2))

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
import torch
from diffusers.models.attention_processor import Attention
from typing import Optional
import dfav2
from ditfastattn_api.modules.ilp import solve_ip

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

        attention_first_half = dfav2.headwise_arrow_attn(query_first_half.transpose(1,2), 
                                                               key_first_half.transpose(1,2), 
                                                               value_first_half.transpose(1,2), 
                                                               window_sizes=window_sizes[head_mask,:],
                                                               seqlen_q_vision = S - 333, 
                                                               seqlen_k_vision = S - 333
                                                              )
        attention_second_half = attention_first_half.clone()
        attention_replace = torch.cat([attention_first_half, attention_second_half], dim=0)  # shape: (B, S, H_replace, D)
    else:
        attention_replace = torch.empty(B, S, 0, D, device=query.device)  # 如果没有需要替换的 head，返回空张量

    # 对于不需要替换的 head，正常计算所有 batch 的 attention
    if query_normal.size(2) > 0:  # 如果有不需要替换的 head
        attention_first_half = dfav2.headwise_arrow_attn(query_first_half.transpose(1,2), 
                                                               key_first_half.transpose(1,2), 
                                                               value_first_half.transpose(1,2), 
                                                               window_sizes=window_sizes[~head_mask,:],
                                                               seqlen_q_vision = S - 333, 
                                                               seqlen_k_vision = S - 333
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


class MMDiTFastAttnProcessor:
    def __init__(self, steps_method=None, alpha=0, window_func=None):
        self.window_func = window_func
        self.steps_method = steps_method
        self.forward_mode = "normal" # could be "normal" or "perhead_normal" or "calib_get_info"
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
        
    
    def run_forward(self, attn:Attention, hidden_states, encoder_hidden_states, qkv_process_func):
        
        forward_args = dict(attn=attn, hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states)
        residual = hidden_states
        batch_size = hidden_states.shape[0]

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
        
        # `context` projections.
        if encoder_hidden_states is not None:
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

            query = torch.cat([query, encoder_hidden_states_query_proj], dim=2)
            key = torch.cat([key, encoder_hidden_states_key_proj], dim=2)
            value = torch.cat([value, encoder_hidden_states_value_proj], dim=2)


        hidden_states = qkv_process_func(query, key, value, forward_args)
        if self.forward_mode == "calib_collect_info":
            self.prev_calib_output = hidden_states
        
        hidden_states = hidden_states.reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            # Split the attention outputs.
            hidden_states, encoder_hidden_states = (
                hidden_states[:, : residual.shape[1]],
                hidden_states[:, residual.shape[1] :],
            )
            if not attn.context_pre_only:
                encoder_hidden_states = attn.to_add_out(encoder_hidden_states)
        
        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)
        
        if encoder_hidden_states is not None:
            return hidden_states, encoder_hidden_states
        else:
            return hidden_states

    
    def calib_qkv_process_func(self, query, key, value, forward_args):
        headwise_relative_MSE={} 
        attn=forward_args["attn"]
        candidates = self.dfa_config.get_available_candidates(attn.name)
        B, H, S, _ = query.shape
        full_hidden_states = dfav2.flash_attn_func(query.transpose(1,2), 
                                                             key.transpose(1,2), 
                                                             value.transpose(1,2))
        for candidate in candidates:
            if "arrow_attn" in candidate:
                # without residual share
                window_size_factor = int(candidate.split("_")[-1])
                window_size = (S - 333) // (window_size_factor * 2)
                hidden_states = dfav2.headwise_arrow_attn(query.transpose(1,2), 
                                                                    key.transpose(1,2), 
                                                                    value.transpose(1,2), 
                                                                    window_sizes=torch.ones((H, 2), device=query.device, dtype=torch.int32) * window_size, 
                                                                    seqlen_q_vision = S - 333, 
                                                                    seqlen_k_vision = S - 333)
            # output share
            elif candidate == 'output_share':
                hidden_states = self.prev_calib_output

            # add cfg sharing
            if "cfg_share" in candidate:
                hidden_states = hidden_states[:B//2].repeat(2, 1, 1, 1)

            rse = (((hidden_states - full_hidden_states)**2).mean(dim=(0,1,3)) / ((full_hidden_states - full_hidden_states.mean(dim=(0,1,3), keepdim=True))**2).mean(dim=(0,1,3))).detach().cpu()
            for h in range(rse.size()[0]):
                if rse[h].item() < self.alpha / H * 1.5:
                    headwise_relative_MSE[(h, candidate)] = rse[h].item()

        # cache compression plan
        if self.stepi not in self.dfa_config.plan.keys():
            self.dfa_config.plan[self.stepi] = {}
        if bool(headwise_relative_MSE):
            head_method_list = solve_ip(headwise_relative_MSE,self.dfa_config.latency, self.alpha)
            print(head_method_list)
        else:
            head_method_list = {}
        self.dfa_config.plan[self.stepi] = head_method_list

        # per head, gen kernel
        if "cfg_share" not in candidates:
            wt = -torch.ones(24, dtype=torch.int32, device=query.device)
            self.output_share_dict[self.stepi] = []
            for head_method in head_method_list:
                head = head_method[0]
                ws = head_method[1]
                if ws == "0":
                    wt[head] = 0
                    self.output_share_dict[self.stepi].append(head)
                else:
                    wt[head] = (S - 333) // (int(ws)*2)
            wt = wt.repeat_interleave(2, dim=0).view(-1, 2)
            self.wt[self.stepi] = wt

            output = dfav2.headwise_arrow_attn(
                query.transpose(1, 2), 
                key.transpose(1, 2), 
                value.transpose(1, 2), 
                window_sizes=wt,
                seqlen_q_vision = S - 333, 
                seqlen_k_vision = S - 333
            )
        else:
            self.asc_enable = True
            wt = -torch.ones(24, device=query.device, dtype=torch.int32)
            cfg_mask = [False] * 24
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
                    wt[head] = (S - 333) // abs(int(ws)*2)
            self.wt[self.stepi] = wt
            if self.cfg_mask is None:
                self.cfg_mask = {}
            self.cfg_mask[self.stepi] = cfg_mask
            
            if any(cfg_mask):
                output = headwise_cfg_attention(query, key, value, cfg_mask, self.timestep_block_mask[self.stepi])
            else:
                output = dfav2.headwise_arrow_attn(
                    query.transpose(1, 2), 
                    key.transpose(1, 2), 
                    value.transpose(1, 2), 
                    window_sizes=wt,
                    seqlen_q_vision = S - 333, 
                    seqlen_k_vision = S - 333
                )
        # B S H D
        output[:,:,self.output_share_dict[self.stepi],:] = self.prev_calib_output[:,:,self.output_share_dict[self.stepi],:]
        return output
    
    
    def raw_qkv_process_func(self, query, key, value, forward_args):
        hidden_states = dfav2.flash_attn_func(query.transpose(1,2), 
                                                        key.transpose(1,2), 
                                                        value.transpose(1,2))
        return hidden_states
    

    def qkv_process_perhead_func(self, query, key, value, forward_args):
        _, _, S, _ = query.shape
        
        if self.asc_enable:
            if self.timestep_block_mask[self.stepi][1] is not None:
                output = headwise_cfg_attention(query, key, value, self.cfg_mask[self.stepi], self.wt[self.stepi])
            else:
                output = dfav2.headwise_arrow_attn(
                    query.transpose(1,2), 
                    key.transpose(1,2), 
                    value.transpose(1,2), 
                    window_sizes = self.wt[self.stepi],
                    seqlen_q_vision = S - 333, 
                    seqlen_k_vision = S - 333
                )
        else:
            output = dfav2.headwise_arrow_attn(
                    query.transpose(1,2), 
                    key.transpose(1,2), 
                    value.transpose(1,2), 
                    window_sizes = self.wt[self.stepi],
                    seqlen_q_vision = S - 333, 
                    seqlen_k_vision = S - 333
                )
        # if output[:,:,self.output_share_dict[self.stepi],:].shape != self.cached_output.shape:
        #     breakpoint()
        output[:,:,self.output_share_dict[self.stepi],:] = self.cached_output
        if self.stepi+1 in self.output_share_dict.keys():
            self.cached_output = output[:,:,self.output_share_dict[self.stepi+1],:]
        return output
    
    def raw_qkv_process_after_calib_func(self, query, key, value, forward_args):
        hidden_states = dfav2.flash_attn_func(query.transpose(1,2), 
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
        temb: Optional[torch.FloatTensor] = None,
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
        
        if encoder_hidden_states is not None:
            hidden_states, encoder_hidden_states = self.run_forward(attn, hidden_states, encoder_hidden_states, qkv_process_func)
            return hidden_states, encoder_hidden_states
        else:
            hidden_states = self.run_forward(attn, hidden_states, encoder_hidden_states, qkv_process_func)
            return hidden_states
        
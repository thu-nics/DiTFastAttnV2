import torch.nn as nn
import torch
import copy
from time import time


class DiTFastAttnFFN(nn.Module):
    def __init__(self, raw_module, steps_method=None, cond_first=None, alpha=0):
        super().__init__()
        self.raw_module = raw_module
        self.steps_method = steps_method
        self.stepi = 0
        self.cache_output = None
        self.mode = "normal" # normal, get_calib_info
        self.dfa_config=None
        self.alpha = alpha
        self.cond_first = cond_first
        self.need_cache_output = True

    def update_config(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, copy.deepcopy(value))

    def calibration(self, hidden_states):
        # breakpoint()
        # first step 
        if self.stepi == 0:
            out = self.raw_module(hidden_states)
            # save for next step inference
            self.cache_output = out
        else:
            raw_out = self.raw_module(hidden_states)
            out = raw_out
            candidates = self.dfa_config.get_available_candidates(self.name)
            for candidate in candidates:
                if candidate == "output_share":
                    curr_out = self.cache_output
                elif candidate == "cfg_share":
                    batch_size = hidden_states.shape[0]
                    if self.cond_first:
                        condition_out = raw_out[:(batch_size // 2)]
                        curr_out = torch.cat([condition_out, condition_out], dim=0)
                    else:
                        condition_out = raw_out[(batch_size // 2):]
                    curr_out = torch.cat([condition_out, condition_out], dim=0)
                rse = (((curr_out - raw_out)**2).mean() / ((raw_out - raw_out.mean())**2).mean()).detach().cpu()
                print(f"{self.name} {candidate}, {rse}")
                if rse < 0.05:
                    self.steps_method[self.stepi] = candidate
                    if self.stepi not in self.dfa_config.ffn_plan.keys():
                        self.dfa_config.ffn_plan[self.stepi] = {}
                    self.dfa_config.ffn_plan[self.stepi][self.name] = candidate
                    out = curr_out
        return out

    def forward(self, hidden_states):
        if self.mode == "get_calib_info":
            out = self.calibration(hidden_states)
        else:
            method = self.steps_method[self.stepi]
            if method == "output_share":
                out = self.cache_output
            elif "cfg_attn_share" in method:
                batch_size = hidden_states.shape[0]
                if self.cond_first:
                    x = hidden_states[: batch_size // 2]
                else:
                    x = hidden_states[batch_size // 2 :]
                out = self.raw_module(x)
                out = torch.cat([out, out], dim=0)
            else:
                out = self.raw_module(hidden_states)
            if self.need_cache_output:
                self.cache_output = out
        return out
    
    def get_latency(self, hidden_states, iter = 100):
        st = time()
        for i in range(iter):
            self.forward(hidden_states)
            torch.cuda.synchronize()
        et = time()
        return et - st


class ForaFFN(nn.Module):
    def __init__(self, raw_module, steps_method=None, cond_first=None, alpha=0):
        super().__init__()
        self.raw_module = raw_module
        self.steps_method = steps_method
        self.stepi = 0
        self.cache_output = None
        self.mode = "normal" # normal, get_calib_info
        self.dfa_config=None
        self.alpha = alpha
        self.cond_first = cond_first
        self.need_cache_output = True

    def update_config(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, copy.deepcopy(value))

    def forward(self, hidden_states):
        print(self.stepi)
        if self.stepi % 2 == 0:
            out = self.raw_module(hidden_states)
            self.cache_output = out
        else:
            out = self.cache_output
        return out
    
    def get_latency(self, hidden_states, iter = 100):
        st = time()
        for i in range(iter):
            self.forward(hidden_states)
            torch.cuda.synchronize()
        et = time()
        return et - st
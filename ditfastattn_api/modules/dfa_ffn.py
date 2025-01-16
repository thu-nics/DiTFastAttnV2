import torch.nn as nn
import torch
import copy
from time import time


class DiTFastAttnFFN(nn.Module):
    def __init__(self, raw_module, steps_method=None, cond_first=None):
        super().__init__()
        self.raw_module = raw_module
        self.steps_method = steps_method
        self.stepi = 0
        self.cache_output = None

        self.cond_first = cond_first
        self.need_cache_output = True

    def update_config(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, copy.deepcopy(value))

    def forward(self, hidden_states):
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
            self.cache_output = out.detach()
        return out
    
    def get_latency(self, hidden_states, iter = 100):
        st = time()
        for i in range(iter):
            self.forward(hidden_states)
            torch.cuda.synchronize()
        et = time()
        return et - st

import torch.nn as nn
import torch


class FastFeedForward(nn.Module):
    def __init__(self, raw_module, steps_method, cond_first):
        super().__init__()
        self.raw_module = raw_module
        self.steps_method = steps_method
        self.stepi = None
        self.cache_output = None

        self.cond_first = cond_first
        self.need_cache_output = True

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
            self.cache_output = out
        self.stepi += 1
        return out

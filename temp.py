from torch.nn.attention.flex_attention import _mask_mod_signature, or_masks
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
from functools import partial
from time import time
import torch
import math
from flash_attn import flash_attn_func

def generate_partial_sliding_window(window_size: int, vtok_len: int) -> _mask_mod_signature:
    """Generates a sliding window attention mask with a given window size.
    Args:
        window_size: The size of the sliding window.

    Note:
        We assume that the window size represents the lookback size and we mask out all future tokens
        similar to causal masking.
    """

    def sliding_window_with_offset(b, h, q_idx, kv_idx):
        return torch.abs(q_idx - kv_idx) <= window_size // 2 | (q_idx >= vtok_len) | (kv_idx >= vtok_len)

    sliding_window_mask = sliding_window_with_offset
    sliding_window_mask.__name__ = f"sliding_window_{window_size}"
    return sliding_window_mask

def flex_attn_wrapper(q, k, v):
    return flex_attention(q, k, v)

def flex_attn_block_mask_wrapper(q, k, v, block_mask):
    return flex_attention(q, k, v, block_mask=block_mask)

flex_attn_compiled = torch.compile(flex_attn_wrapper, dynamic=False)
flex_attn_block_mask_compiled = torch.compile(flex_attn_block_mask_wrapper, dynamic=False)

# flex attn partial window attention
# def generate_partial_sliding_window(window_size: int, vtok_len: int) -> _mask_mod_signature:
#     """Generates a sliding window attention mask with a given window size.
#     Args:
#         window_size: The size of the sliding window.

#     Note:
#         We assume that the window size represents the lookback size and we mask out all future tokens
#         similar to causal masking.
#     """

#     def sliding_window_with_offset(b, h, q_idx, kv_idx):
#         return torch.abs(q_idx - kv_idx) <= window_size // 2 
    
#     def full(b, h, q_idx, kv_idx):
#         return (q_idx >= vtok_len) | (kv_idx >= vtok_len)

#     sliding_window_mask = or_masks(sliding_window_with_offset, full)
#     sliding_window_mask.__name__ = f"sliding_window_{window_size}"
#     return sliding_window_mask

def causal_mask(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx




B = 6
S = 4096 + 333
H = 24
D = 64
bs = int(math.sqrt(S - 333))
query = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16)
key = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16)
value = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16)

print("done")

st = time()
for i in range(20):
    torch.cuda.synchronize()
    for j in range(100):
        output = flash_attn_func(query, key, value)
        torch.cuda.synchronize()

et = time()

print(f"100 iteration time: {(et - st) / 20}s")
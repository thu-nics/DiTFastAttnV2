from torch.nn.attention.flex_attention import _mask_mod_signature, or_masks
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
from functools import partial
from time import time
import torch
import math

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

def generate_partial_sliding_block_window(window_size: int, vtok_len: int, block_size=64) -> _mask_mod_signature:
    """Generates a sliding window attention mask with a given window size.
    Args:
        window_size: The size of the sliding window.

    Note:
        We assume that the window size represents the lookback size and we mask out all future tokens
        similar to causal masking.
    """

    def sliding_window_with_offset(b, h, q_idx, kv_idx):
        blockq_idx = q_idx // block_size * block_size
        return ((kv_idx >= blockq_idx - window_size // 2 * block_size) & (kv_idx <= blockq_idx + (window_size // 2 + 1) * block_size)) | (q_idx >= vtok_len) | (kv_idx >= vtok_len)

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

def transform_dense_block_mask(mask, block_size=(64, 64)):
    _, nhead, H, W = mask.shape
    _H = H // block_size[0] * block_size[0]
    _W = W // block_size[1] * block_size[1]
    _mask = mask[:, :_H, :_W].view(nhead, H // block_size[0], block_size[0], W // block_size[1], block_size[1])
    _mask_max = _mask.amax(dim=[2, 4], keepdim=True)
    _mask |= _mask_max
    mask[:, :_H, :_W] = _mask.reshape(nhead, _H, _W)
    return mask

def causal_mask(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx




B = 6
S = 4096 + 333
H = 24
D = 64
bs = int(math.sqrt(S - 333))
query = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
key = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
value = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)

# torch._dynamo.config.cache_size_limit = 16

# for window_size in [128, 256, 512]:
#     sliding_window_mask = generate_partial_sliding_window(window_size=window_size, vtok_len=S - 333)
#     for block_size in [64, 128, 256]:
#         block_mask = create_block_mask(sliding_window_mask, None, None, S, S, device="cuda", _compile=True, BLOCK_SIZE=block_size)
#         # print(block_mask.shape)
#         # print(block_mask)
#         st = time()
#         for i in range(10):
#             torch.cuda.synchronize()
#             for j in range(100):
#                 output = flex_attn_block_mask_compiled(query, key, value, block_mask=block_mask)
#                 # output = flex_attn_compiled(query, key, value)
#             torch.cuda.synchronize()

#         et = time()
#         print(f"window_size: {window_size}, block_size: {block_size}, 100 iteration time: {(et - st) / 10}s")


sliding_window_mask = generate_partial_sliding_window(window_size=(S - 333) // 8, vtok_len=S - 333)
block_mask = create_block_mask(sliding_window_mask, None, None, S, S, device="cuda", _compile=True, BLOCK_SIZE=128)
print(block_mask.shape)
print(block_mask)
# block_mask = create_block_mask(causal_mask, B, H, S, S, device="cuda", _compile=True)

st = time()
for i in range(20):
    torch.cuda.synchronize()
    for j in range(100):
        output = flex_attn_block_mask_compiled(query, key, value, block_mask=block_mask)
        # output = flex_attn_compiled(query, key, value)
        torch.cuda.synchronize()

et = time()

print(f"100 iteration time: {(et - st) / 20}s")

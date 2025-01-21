import os,sys
sys.path.append('.')
import torch
import torch.nn.functional as F
from ditfastattn_api.blockwise_window_attn import blockwise_window_mm_attn

def debug_blockwise_window_attn_no_text():
    bs=1
    nhead=2
    query_len=kv_len=64
    txt_len=0
    block_size=16
    head_dim=16
    q=torch.ones(bs,nhead,query_len,head_dim).cuda().half()
    k=torch.ones(bs,nhead,kv_len,head_dim).cuda().half()
    v=torch.ones(bs,nhead,kv_len,head_dim).cuda().half()
    v*=torch.arange(kv_len).view(1,1,kv_len,1).cuda().half()
    left_window_size=torch.ones(nhead,dtype=torch.int32).cuda()*16
    left_window_size[1]=32
    right_window_size=torch.ones(nhead,dtype=torch.int32).cuda()*16
    right_window_size[1]=16
    
    o=blockwise_window_mm_attn(q,k,v,query_len,txt_len,left_window_size,right_window_size,block_size)
    print(o[:,:,:,0])

def accuracy_blockwise_window_attn_no_text():
    bs=2
    nhead=16
    query_len=kv_len=32
    txt_len=0
    block_size=16
    head_dim=16
    dtype=torch.float32
    q=torch.randn(bs,nhead,query_len,head_dim).cuda().to(dtype)
    k=torch.randn(bs,nhead,kv_len,head_dim).cuda().to(dtype)
    v=torch.randn(bs,nhead,kv_len,head_dim).cuda().to(dtype)
    v*=torch.arange(kv_len).view(1,1,kv_len,1).cuda().to(dtype)
    left_window_size=torch.ones(nhead,dtype=torch.int32).cuda()*16
    right_window_size=torch.ones(nhead,dtype=torch.int32).cuda()*16
    
    attention_mask=torch.zeros(query_len,kv_len,dtype=torch.bool).cuda()
    attention_mask[:16,:16]=True
    attention_mask[16:,:]=True
    
    o=blockwise_window_mm_attn(q,k,v,query_len,txt_len,left_window_size,right_window_size,block_size)
    raw_o=F.scaled_dot_product_attention(q,k,v,attn_mask=attention_mask)
    print(o-raw_o)
    print(o)
    print(raw_o)

    
if __name__=='__main__':
    # debug_blockwise_window_attn_no_text()
    accuracy_blockwise_window_attn_no_text()
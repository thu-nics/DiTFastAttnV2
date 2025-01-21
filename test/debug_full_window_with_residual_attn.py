import os,sys
sys.path.append('.')
import torch
import torch.nn.functional as F
from ditfastattn_api.full_window_attn_with_window_residual import full_mm_attn_with_window_residual

    
def debug_blockwise_window_attn_with_text():
    bs=1
    nhead=1
    query_len=kv_len=32
    txt_len=20
    total_len=query_len+txt_len
    block_size=16
    head_dim=256
    q=torch.ones(bs,nhead,total_len,head_dim).cuda().half()
    k=torch.ones(bs,nhead,total_len,head_dim).cuda().half()
    v=torch.ones(bs,nhead,total_len,head_dim).cuda().half()
    v*=torch.arange(total_len).view(1,1,total_len,1).cuda().half()
    left_window_size=torch.ones(nhead,dtype=torch.int32).cuda()*16
    # left_window_size[1]=32
    right_window_size=torch.ones(nhead,dtype=torch.int32).cuda()*16
    # right_window_size[1]=16
    
    o,residual=full_mm_attn_with_window_residual(q,k,v,query_len,txt_len,left_window_size,right_window_size,block_size)
    print(o[:,:,:,0])
    print(residual[:,:,:,0])
    
def accuracy_blockwise_window_attn_with_text():
    bs=1
    nhead=2
    query_len=kv_len=32
    txt_len=16
    total_len=query_len+txt_len
    block_size=16
    head_dim=16
    dtype=torch.float32
    q=torch.randn(bs,nhead,total_len,head_dim).cuda().to(dtype)
    k=torch.randn(bs,nhead,total_len,head_dim).cuda().to(dtype)
    v=torch.randn(bs,nhead,total_len,head_dim).cuda().to(dtype)
    
    # q=torch.ones(bs,nhead,total_len,head_dim).cuda().to(dtype)
    # k=torch.ones(bs,nhead,total_len,head_dim).cuda().to(dtype)
    # v=torch.ones(bs,nhead,total_len,head_dim).cuda().to(dtype)
    # v*=torch.arange(total_len).view(1,1,total_len,1).cuda().to(dtype)
    left_window_size=torch.ones(nhead,dtype=torch.int32).cuda()*16
    right_window_size=torch.ones(nhead,dtype=torch.int32).cuda()*16
    
    attention_mask=torch.zeros(total_len,total_len,dtype=torch.bool).cuda()
    attention_mask[:16,:16]=True
    attention_mask[:16,-txt_len:]=True
    attention_mask[16:,:]=True
    # print(attention_mask[::8,::8])
    
    o,residual=full_mm_attn_with_window_residual(q,k,v,query_len,txt_len,left_window_size,right_window_size,block_size)
    window_o=o-residual
    raw_o=F.scaled_dot_product_attention(q,k,v,attn_mask=attention_mask)
    print((window_o-raw_o)[:,:,:,0])
    print(window_o[:,:,:,0])
    print(raw_o[:,:,:,0])

    
if __name__=='__main__':
    # debug_blockwise_window_attn_no_text()
    # accuracy_blockwise_window_attn_no_text()
    accuracy_blockwise_window_attn_with_text()
    # debug_blockwise_window_attn_with_text()
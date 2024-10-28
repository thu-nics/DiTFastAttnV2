import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.attention import FeedForward, BasicTransformerBlock
from diffusers.models.attention_processor import Attention, AttnProcessor2_0
from ditfastattn_api.modules.dfa_ffn import DiTFastAttnFFN
from ditfastattn_api.modules.dfa_processor import DiTFastAttnProcessor
from ditfastattn_api.dfa_config import DiTFastAttnConfig
import time


def transform_model_dfa(model, n_steps=20):
    """
    transform the attention and ffn in a transformer model with DitFastAttnProcessor and DitFastAttnFFN.
    Return the dfa config.
    """
    if not isinstance(model, nn.Module):
        raise ValueError("model must be a nn.Module")
    dfa_config = DiTFastAttnConfig()
    for name, module in model.named_modules():
        if isinstance(module, BasicTransformerBlock):
            if isinstance(module.attn1, Attention):
                module.attn1.processor = DiTFastAttnProcessor(steps_method=["raw" for _ in range(n_steps)])
                dfa_config.add_attn_processor(name + ".attn1", module.attn1.processor)
            if isinstance(module.ff, FeedForward):
                module.ff = DiTFastAttnFFN(module.ff, steps_method=["raw" for _ in range(n_steps)])
                dfa_config.add_ffn_layer(name + ".ff", module.ff)

    def refresh_stepi_hook(module, input):
        for name, module in module.named_modules():
            if hasattr(module, "stepi"):
                module.stepi = 0
            if isinstance(module, Attention):
                module.processor.stepi = 0

    model.register_forward_pre_hook(refresh_stepi_hook)
    return dfa_config


def transformer_input_hook(module, arg, kwargs, output):
    module.saved_input = (arg, kwargs)


def dfa_test_latency(pipe, *args, warmup=1, repeat=3, only_transformer=True, **kwargs):
    """
    return latency
    """
    handeler = pipe.transformer.register_forward_hook(transformer_input_hook, with_kwargs=True)
    pipe(*args, **kwargs)
    handeler.remove()

    st = time.time()

    for i in range(repeat):
        if only_transformer:
            pipe.transformer(*pipe.transformer.saved_input[0], **pipe.transformer.saved_input[1])
        else:
            pipe(*args, **kwargs)
    torch.cuda.synchronize()
    ed = time.time()
    t = (ed - st) / repeat
    print(f"average time for inference: {t}")
    return t

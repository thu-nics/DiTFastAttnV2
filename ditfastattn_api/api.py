import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.attention import FeedForward, BasicTransformerBlock, JointTransformerBlock
from diffusers.models.attention_processor import Attention, AttnProcessor2_0
from ditfastattn_api.modules.dfa_ffn import DiTFastAttnFFN
from ditfastattn_api.modules.dfa_processor import DiTFastAttnProcessor, MMDiTFastAttnProcessor
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
            # for DiT
            if isinstance(module.attn1, Attention):
                module.attn1.processor = DiTFastAttnProcessor(steps_method=["raw" for _ in range(n_steps)])
                dfa_config.add_attn_processor(name + ".attn1", module.attn1.processor)
            if isinstance(module.ff, FeedForward):
                module.ff = DiTFastAttnFFN(module.ff, steps_method=["raw" for _ in range(n_steps)])
                dfa_config.add_ffn_layer(name + ".ff", module.ff)
        if isinstance(module, JointTransformerBlock):
            # for SD3
            if isinstance(module.attn, Attention):
                module.attn.processor = MMDiTFastAttnProcessor(steps_method=["raw" for _ in range(n_steps)])
                dfa_config.add_attn_processor(name + ".attn", module.attn.processor)
            if isinstance(module.ff, FeedForward):
                module.ff = DiTFastAttnFFN(module.ff, steps_method=["raw" for _ in range(n_steps)])
                dfa_config.add_ffn_layer(name + ".ff", module.ff)
    return dfa_config


def register_refresh_stepi_hook(model, n_steps):
    def refresh_stepi_hook(module, input, output):
        for name, module in module.named_modules():
            if hasattr(module, "stepi"):
                module.stepi += 1
                if module.stepi == n_steps:
                    module.stepi = 0
                # print(f"stepi of {name} is {module.stepi}")
            if isinstance(module, Attention):
                module.processor.stepi += 1
                if module.processor.stepi == n_steps:
                    module.processor.stepi = 0
                # print(f"stepi of {name}.attn is {module.processor.stepi}")

    hook = model.register_forward_hook(refresh_stepi_hook)
    model.refresh_stepi_hook = hook

    # reset the stepi of all the attention and ffn layers
    for name, module in model.named_modules():
        if isinstance(module, Attention):
            module.processor.stepi = 0
        if isinstance(module, DiTFastAttnFFN):
            module.stepi = 0


def unregister_refresh_stepi_hook(model):
    if hasattr(model, "refresh_stepi_hook"):
        model.refresh_stepi_hook.remove()
        delattr(model, "refresh_stepi_hook")


def transformer_input_hook(module, arg, kwargs, output):
    module.saved_input = (arg, kwargs)


def dfa_test_latency(pipe, *args, warmup=1, repeat=3, only_transformer=True, **kwargs):
    """
    return latency
    """
    handeler = pipe.transformer.register_forward_hook(transformer_input_hook, with_kwargs=True)
    pipe(*args, **kwargs)
    handeler.remove()

    def transformer_pre_forward_hook(module, input):
        if module.transformer_step_i == 0:
            torch.cuda.synchronize()
            module.start_time = time.time()

    def transformer_post_forward_hook(module, input, output):
        module.transformer_step_i += 1
        if module.transformer_step_i == kwargs["num_inference_steps"]:
            torch.cuda.synchronize()
            module.end_time = time.time()

    if only_transformer:
        pre_hook_handler = pipe.transformer.register_forward_pre_hook(transformer_pre_forward_hook)
        post_hook_handler = pipe.transformer.register_forward_hook(transformer_post_forward_hook)
        t = 0
        for i in range(repeat):
            pipe.transformer.transformer_step_i = 0
            pipe(*args, **kwargs)
            t += pipe.transformer.end_time - pipe.transformer.start_time
        t = t / repeat
        pre_hook_handler.remove()
        post_hook_handler.remove()
        print(f"average time for transformer inference: {t}")

    else:
        st = time.time()
        for i in range(repeat):
            pipe(*args, **kwargs)
        torch.cuda.synchronize()
        ed = time.time()
        t = (ed - st) / repeat
        print(f"average time for pipeline inference: {t}")
    return t

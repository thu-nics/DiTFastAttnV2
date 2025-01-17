import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.attention import FeedForward, BasicTransformerBlock, JointTransformerBlock
from diffusers.models.attention_processor import Attention, AttnProcessor2_0
from ditfastattn_api.modules.dfa_ffn import DiTFastAttnFFN
# from ditfastattn_api.modules.dfa_processor import DiTFastAttnProcessor, MMDiTFastAttnProcessor
# --------
from ditfastattn_api.modules.dfa_processor import DiTFastAttnProcessor
from ditfastattn_api.modules.dfa_processor_revised import MMDiTFastAttnProcessor
# --------
from ditfastattn_api.dfa_config import DiTFastAttnConfig
import time
import numpy as np

attn_class = Attention

def transform_model_dfa(model, n_steps=20, window_func=None):
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
                module.attn.processor = MMDiTFastAttnProcessor(steps_method=["raw" for _ in range(n_steps)], window_func=window_func)
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

def dfa_test_layer_latency(pipe, n_steps, *args, **kwargs):
    """
    return latency of different method
    """
    attn_latency = []
    ffn_latency = []

    # method 

    def module_pre_forward_hook(m, input):
        if isinstance(m, DiTFastAttnFFN):
            if m.timestep_index != 0:
                torch.cuda.synchronize()
                m.st = time.time()
        if isinstance(m, attn_class):
            if m.timestep_index != 0:
                torch.cuda.synchronize()
                m.st = time.time()

    def module_post_forward_hook(m, input, output):
        if isinstance(m, DiTFastAttnFFN):
            if m.timestep_index != 0:    
                torch.cuda.synchronize()
                m.et = time.time()
                ffn_latency.append(m.et - m.st)
            m.timestep_index += 1
            
        if isinstance(m, attn_class):
            if m.timestep_index != 0:
                torch.cuda.synchronize()
                m.et = time.time()
                attn_latency.append(m.et - m.st)
            m.timestep_index += 1

    all_hooks = []
    for name, module in pipe.transformer.named_modules():
        module.name = name
        if isinstance(module, JointTransformerBlock):
            # for MMDiT
            if isinstance(module.attn, attn_class):
                module.attn.timestep_index = 0
                module.attn.processor.compression_influences = {}
                hook = module.attn.register_forward_pre_hook(module_pre_forward_hook)
                all_hooks.append(hook)
                hook = module.attn.register_forward_hook(module_post_forward_hook)
                all_hooks.append(hook)
            if isinstance(module.ff, DiTFastAttnFFN):
                module.ff.timestep_index = 0
                module.ff.compression_influences = {}
                hook = module.ff.register_forward_pre_hook(module_pre_forward_hook)
                all_hooks.append(hook)
                hook = module.ff.register_forward_hook(module_post_forward_hook)
                all_hooks.append(hook)

    register_refresh_stepi_hook(pipe.transformer, n_steps=n_steps)
    pipe(*args, **kwargs)
    unregister_refresh_stepi_hook(pipe.transformer)

    # remove hooks
    for hook in all_hooks:
        hook.remove()
    
    return np.mean(attn_latency), np.mean(ffn_latency)

        
class MethodSpeedup:
    def __init__(self, vtok_len = 4096, ttok_len = 333):
        # examples of candidate: ("attn", "residual_window_attn"), ("ff", "output_share")
        self.candidates = []
        self.speedup_dict = {('attn','raw'):1, ('ff', 'raw'):1}
        self.vtok_len = vtok_len
        self.ttok_len = ttok_len
    
    def load_candidates(self, candidates):
        for candidate in candidates:
            self.candidates.append(candidate)
            # self.speedup_dict[candidate] = 0
    
    def generate_latency(self, mode='estimate', pipe=None, n_steps=20, dfa_config=None, *args, **kwargs):
        if mode == "estimate":
            for candidate in self.candidates:
                print(candidate)
                if candidate[1] == "output_share":
                    print("It is output_share")
                    self.speedup_dict[candidate] = 1
                elif candidate[1] == "cfg_share":
                    print("It is cfg_share")
                    self.speedup_dict[candidate] = 0.5
                elif "window_attn" in candidate[1]:
                    print("It is window attn")
                    window_size = self.vtok_len // 8
                    full_computation = (self.vtok_len + self.ttok_len)**2
                    ratio = (full_computation -  (self.vtok_len - window_size // 2) **2) / full_computation 
                    if "cfg_attn_share" in candidate:
                        ratio = ratio / 2
                    self.speedup_dict[candidate] = 1 - ratio
        elif mode == "test":
            # TODO: identify model type
            block_class = JointTransformerBlock
            attn_class = Attention
            ffn_class = FeedForward
            # latency for raw
            for name, module in pipe.transformer.named_modules():
                module.name = name
                if isinstance(module, block_class):
                    # for MMDiT
                    if isinstance(module.attn, attn_class):
                        module.attn.processor.need_cache_output = False
                        module.attn.processor.cache_residual_forced = False
                        module.attn.processor.calib_mode = "off"
                    if isinstance(module.ff, DiTFastAttnFFN):
                        module.ff.need_cache_output = False

            attn_latency, ffn_latency = dfa_test_layer_latency(pipe, n_steps, *args, **kwargs)
            self.speedup_dict[('attn', 'raw')] = attn_latency
            self.speedup_dict[('ff', 'raw')] = ffn_latency

            for name, module in pipe.transformer.named_modules():
                module.name = name
                if isinstance(module, block_class):
                    # for MMDiT
                    if isinstance(module.attn, attn_class):
                        module.attn.processor.need_cache_output = True
                    if isinstance(module.ff, DiTFastAttnFFN):
                        module.ff.need_cache_output = True

            total_step = len(dfa_config.layers[dfa_config.layer_names[0]]['kwargs']['steps_method'])
            for candidate in self.candidates:
                for layer_name in dfa_config.layers.keys():
                    for step_idx in range(1, total_step):
                        # dfa_test_layer_latency()
                        dfa_config.set_layer_step_method(layer_name, step_idx, candidate)
                attn_latency, ffn_latency = dfa_test_layer_latency(pipe, n_steps, *args, **kwargs)
                self.speedup_dict[('attn', candidate)] = attn_latency
                if 'window_attn' not in candidate:
                    self.speedup_dict[('ff', candidate)] = ffn_latency

            for step_idx in range(1, total_step):
                dfa_config.reset_step_method(step_idx)
        latency_dict = self.speedup_dict
        latency = {}
        raw_attn_lattency = latency_dict[('attn','raw')].item()
        raw_ff_lattency = latency_dict[('ff', 'raw')].item()
        for layer_type, method in latency_dict.keys():
            print(f"layer type: {layer_type}, method: {method}")
            if method != "raw":
                if layer_type == "attn":
                    latency[(layer_type, method)] = raw_attn_lattency - latency_dict[(layer_type, method)].item()
                elif layer_type == "ff":
                    latency[(layer_type, method)] = raw_ff_lattency - latency_dict[(layer_type, method)].item()
        return latency
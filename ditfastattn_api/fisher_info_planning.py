from diffusers.models.attention import FeedForward, BasicTransformerBlock, JointTransformerBlock
from diffusers.models.attention_processor import Attention, AttnProcessor2_0
from ditfastattn_api.modules.dfa_ffn import DiTFastAttnFFN
import torch

def get_layer_fisher_info(pipe, dataloader, model_misc):
    layer_gradients = {}

    def collect_out_grad_backward_hook(m, grad_input, grad_output):
        if m.timestep_index not in layer_gradients:
            layer_gradients[m.timestep_index] = {}
        if m.name not in layer_gradients[m.timestep_index]:
            layer_gradients[m.timestep_index][m.name] = 0
        layer_gradients[m.timestep_index][m.name] += grad_output[0].sum(0)  # sum over batch size
        # grad_output[0]

    all_hooks = []
    for name, module in pipe.transformer.named_modules():
        module.name = name
        if isinstance(module, model_misc.block_class):
            # for DiT
            if isinstance(module.attn1, model_misc.attn_class):
                hook = module.attn1.register_full_backward_hook(collect_out_grad_backward_hook)
                all_hooks.append(hook)
            if isinstance(module.ff, model_misc.ffn_class):
                hook = module.ff.register_full_backward_hook(collect_out_grad_backward_hook)
                all_hooks.append(hook)

    for i, (args, kwargs) in enumerate(dataloader()):
        print(f">>> calibration fisher info sample {i} <<<")
        model_misc.inference_fn_with_backward(pipe, *args, **kwargs)

    # remove hooks
    for hook in all_hooks:
        hook.remove()

    layer_fisher_info = {}
    for timestep_index, step_gradients in layer_gradients.items():
        layer_fisher_info[timestep_index] = {}
        for layer_name, gradient in step_gradients.items():
            # layer_fisher_info[timestep_index][layer_name] = gradient.pow(2)
            layer_fisher_info[timestep_index][layer_name] = gradient

    return layer_fisher_info


def get_compression_method_influence(pipe, dfa_config, dataloader, layer_fisher_info, model_misc):
    layer_compression_influences = {}

    def collect_compression_influence_hook(m, input, kwargs, output):
        candidates = dfa_config.get_available_candidates(m.name)
        if m.timestep_index != 0:
            # the first step cannot take use of the preivous step's output
            for candidate in candidates:
                dfa_config.set_layer_step_method(m.name, m.timestep_index, candidate)
                new_output = m.forward(*input, **kwargs).float()
                fisher_info = layer_fisher_info[m.timestep_index][m.name]
                # influence = ((new_output - output.float()).sum(0).pow(2) * fisher_info).sum()
                influence = ((new_output - output.float()) * fisher_info).abs().sum()
                if m.name not in layer_compression_influences:
                    layer_compression_influences[m.name] = {}
                if m.timestep_index not in layer_compression_influences[m.name]:
                    layer_compression_influences[m.name][m.timestep_index] = {}
                if candidate not in layer_compression_influences[m.name][m.timestep_index]:
                    layer_compression_influences[m.name][m.timestep_index][candidate] = 0
                layer_compression_influences[m.name][m.timestep_index][candidate] += influence
                # print(f"time {m.timestep_index} layer {m.name} candidate {candidate} influence {influence}")
                dfa_config.set_layer_step_method(m.name, m.timestep_index, "raw")
        else:
            pass

        # manually cache the output for DiTFastAttnFFN and DiTFastAttnProcessor
        if isinstance(m, DiTFastAttnFFN):
            m.cache_output = output
        elif isinstance(m, model_misc.attn_class):
            m.processor.cached_output = output

    all_hooks = []
    for name, module in pipe.transformer.named_modules():
        module.name = name
        if isinstance(module, model_misc.block_class):
            # for DiT
            if isinstance(module.attn1, model_misc.attn_class):
                hook = module.attn1.register_forward_hook(collect_compression_influence_hook, with_kwargs=True)
                all_hooks.append(hook)
                module.attn1.processor.need_cache_output = False
            if isinstance(module.ff, DiTFastAttnFFN):
                hook = module.ff.register_forward_hook(collect_compression_influence_hook, with_kwargs=True)
                all_hooks.append(hook)
                module.ff.need_cache_output = False
    for i, (args, kwargs) in enumerate(dataloader()):
        print(f">>> compression influence sample {i} <<<")
        model_misc.inference_with_timeinfo(pipe, *args, **kwargs)

    # remove hooks
    for hook in all_hooks:
        hook.remove()

    for name, module in pipe.transformer.named_modules():
        if isinstance(module, model_misc.block_class):
            if isinstance(module.attn1, model_misc.attn_class):
                module.attn1.processor.need_cache_output = True
            if isinstance(module.ff, DiTFastAttnFFN):
                module.ff.need_cache_output = True

    return layer_compression_influences


def fisher_info_planning(layer_compression_influences, dfa_config, threshold=0.8):
    # sort all of the layer_compression_influences by the influence
    sorted_layer_compression_influences = []
    for layer_name, step_influences in layer_compression_influences.items():
        for step_index, influence_dict in step_influences.items():
            for candidate, influence in influence_dict.items():
                sorted_layer_compression_influences.append((layer_name, step_index, candidate, influence))
    sorted_layer_compression_influences.sort(key=lambda x: x[3])

    # apply the sorted layer_compression_influences to the dfa_config
    n_compress = int(len(sorted_layer_compression_influences) * threshold)
    compress_methods = {}
    for layer_name, step_index, candidate, influence in sorted_layer_compression_influences[:n_compress]:
        if (layer_name, step_index) not in compress_methods:
            compress_methods[(layer_name, step_index)] = candidate
        else:
            # TODO: set according to the cost of the candidate method
            compress_methods[(layer_name, step_index)] = candidate
    for (layer_name, step_index), candidate in compress_methods.items():
        dfa_config.set_layer_step_method(layer_name, step_index, candidate)
    print(dfa_config)
    return compress_methods


def get_layer_influence(pipe, dataloader, dfa_config, model_misc):
    layer_compression_influences = {}

    def collect_in_out_forward_hook(m, input, kwargs, output):
        if isinstance(m, DiTFastAttnFFN):
            m.cache_output1 = output
            m.cache_input = input
            m.cache_kwargs = kwargs
        elif isinstance(m, model_misc.attn_class):
            m.processor.cached_output1 = output
            m.processor.cached_input = input
            m.processor.cached_kwargs = kwargs
        

    def collect_influence_backward_hook(m, grad_input, grad_output):
        if m.timestep_index != 0:
            candidates = dfa_config.get_available_candidates(m.name)
            if isinstance(m, DiTFastAttnFFN):
                output = m.cache_output1
                input = m.cache_input
                kwargs = m.cache_kwargs
            elif isinstance(m, model_misc.attn_class):
                output = m.processor.cached_output1
                input = m.processor.cached_input
                kwargs = m.processor.cached_kwargs
            for candidate in candidates:
                dfa_config.set_layer_step_method(m.name, m.timestep_index, candidate)
                with torch.no_grad():
                    new_output = m.forward(*input, **kwargs).float()
                influence = ((new_output - output.float()) * grad_output[0]).abs().sum()
                if m.name not in layer_compression_influences:
                    layer_compression_influences[m.name] = {}
                if m.timestep_index not in layer_compression_influences[m.name]:
                    layer_compression_influences[m.name][m.timestep_index] = {}
                if candidate not in layer_compression_influences[m.name][m.timestep_index]:
                    layer_compression_influences[m.name][m.timestep_index][candidate] = 0
                layer_compression_influences[m.name][m.timestep_index][candidate] += influence
                # print(f"time {m.timestep_index} layer {m.name} candidate {candidate} influence {influence}")
                dfa_config.set_layer_step_method(m.name, m.timestep_index, "raw")
        
        if isinstance(m, DiTFastAttnFFN):
            m.cache_output = m.cache_output1
        elif isinstance(m, model_misc.attn_class):
            m.processor.cached_output = m.processor.cached_output1
        
        
            
    all_hooks = []
    for name, module in pipe.transformer.named_modules():
        module.name = name
        if isinstance(module, model_misc.block_class):
            # for DiT
            if isinstance(module.attn1, model_misc.attn_class):
                hook = module.attn1.register_forward_hook(collect_in_out_forward_hook, with_kwargs=True)
                all_hooks.append(hook)
                hook = module.attn1.register_full_backward_hook(collect_influence_backward_hook)
                all_hooks.append(hook)
                module.attn1.processor.need_cache_output = False
            if isinstance(module.ff, DiTFastAttnFFN):
                hook = module.ff.register_forward_hook(collect_in_out_forward_hook, with_kwargs=True)
                all_hooks.append(hook)
                hook = module.ff.register_full_backward_hook(collect_influence_backward_hook)
                all_hooks.append(hook)
                module.ff.need_cache_output = False
    for i, (args, kwargs) in enumerate(dataloader()):
        print(f">>> calibration fisher info sample {i} <<<")
        model_misc.inference_fn_with_backward(pipe, *args, **kwargs)

    # remove hooks
    for hook in all_hooks:
        hook.remove()

    for name, module in pipe.transformer.named_modules():
        module.name = name
        if isinstance(module, model_misc.block_class):
            # for DiT
            if isinstance(module.attn1, model_misc.attn_class):
                module.attn1.processor.need_cache_output = True
                if hasattr(module.attn1.processor, "cached_output1"):
                    del module.attn1.processor.cached_output1
                if hasattr(module.attn1.processor, "cached_input"):
                    del module.attn1.processor.cached_input
                if hasattr(module.attn1.processor, "cached_kwargs"):
                    del module.attn1.processor.cached_kwargs

            if isinstance(module.ff, DiTFastAttnFFN):
                module.ff.need_cache_output = True
                if hasattr(module.ff, "cache_output1"):
                    del module.ff.cache_output1
                if hasattr(module.ff, "cache_input"):
                    del module.ff.cache_input
                if hasattr(module.ff, "cache_kwargs"):
                    del module.ff.cache_kwargs

    return layer_compression_influences

def update_layer_influence_new(pipe, dataloader, dfa_config, model_misc, alpha=1e-8):
    for i, (args, kwargs) in enumerate(dataloader()):
        print(f">>> calibration fisher info sample {i} <<<")
        # model_misc.inference_fn_with_backward_plan_update(pipe, dfa_config, alpha, *args, **kwargs)
        model_misc.inference_fn_with_backward_plan_update_binary(pipe, dfa_config, alpha, *args, **kwargs)
        # model_misc.inference_fn_with_backward_metric_check(pipe, dfa_config, alpha, *args, **kwargs)
    return

def update_layer_influence_two_phase(pipe, dataloader, dfa_config, model_misc, alpha):
    for i, (args, kwargs) in enumerate(dataloader()):
        print(f">>> calibration fisher info sample {i} <<<")
        # breakpoint()
        model_misc.inference_fn_with_backward_plan_update_binary_new_method(pipe, dfa_config, alpha, *args, **kwargs)

def generate_ground_truth(pipe, dataloader, model_misc):
    for i, (args, kwargs) in enumerate(dataloader()):
        print(f">>> calibration fisher info sample {i} <<<")
        # breakpoint()
        model_misc.inference_save_ground_truth(pipe, *args, **kwargs)
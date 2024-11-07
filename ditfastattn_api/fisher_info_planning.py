from diffusers.models.attention import FeedForward, BasicTransformerBlock, JointTransformerBlock
from diffusers.models.attention_processor import Attention, AttnProcessor2_0
from ditfastattn_api.modules.dfa_ffn import DiTFastAttnFFN


def get_layer_fisher_info(pipe, dataloader, model_misc):
    layer_gradients = {}

    def collect_out_grad_backward_hook(m, grad_input, grad_output):
        if m.timestep_index not in layer_gradients:
            layer_gradients[m.timestep_index] = {}
        if m.name not in layer_gradients[m.timestep_index]:
            layer_gradients[m.timestep_index][m.name] = 0
        layer_gradients[m.timestep_index][m.name] += grad_output[0].sum(0)  # sum over batch size

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
            layer_fisher_info[timestep_index][layer_name] = gradient.pow(2)

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
                influence = ((new_output - output.float()).pow(2).sum(0) * fisher_info).sum()
                if m.name not in layer_compression_influences:
                    layer_compression_influences[m.name] = {}
                if m.timestep_index not in layer_compression_influences[m.name]:
                    layer_compression_influences[m.name][m.timestep_index] = {}
                if candidate not in layer_compression_influences[m.name][m.timestep_index]:
                    layer_compression_influences[m.name][m.timestep_index][candidate] = 0
                layer_compression_influences[m.name][m.timestep_index][candidate] += influence
                print(f"time {m.timestep_index} layer {m.name} candidate {candidate} influence {influence}")
                dfa_config.set_layer_step_method(m.name, m.timestep_index, "raw")

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

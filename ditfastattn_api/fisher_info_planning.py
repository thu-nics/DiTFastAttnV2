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

    return layer_gradients


def get_compression_method_influence(pipe, dfa_config, dataloader, layer_fisher_info, model_misc):
    layer_compression_influences = {}

    def collect_compression_influence_hook(m, input, kwargs, output):
        candidates = dfa_config.get_available_candidates(m.name)
        for candidate in candidates:
            dfa_config.set_layer_step_method(m.name, m.timestep_index, candidate)
            new_output = m.forward(*input, **kwargs)
            fisher_info = layer_fisher_info[m.timestep_index][m.name]
            influence = ((new_output - output).pow(2).sum(0) * fisher_info).sum()
            if m.name not in layer_compression_influences:
                layer_compression_influences[m.name] = {}
            if m.timestep_index not in layer_compression_influences[m.name]:
                layer_compression_influences[m.name][m.timestep_index] = {}
            if candidate not in layer_compression_influences[m.name][m.timestep_index]:
                layer_compression_influences[m.name][m.timestep_index][candidate] = 0
            layer_compression_influences[m.name][m.timestep_index][candidate] += influence
            print(f"time {m.timestep_index} layer {m.name} candidate {candidate} influence {influence}")
            dfa_config.set_layer_step_method(m.name, m.timestep_index, "raw")

    all_hooks = []
    for name, module in pipe.transformer.named_modules():
        module.name = name
        if isinstance(module, model_misc.block_class):
            # for DiT
            if isinstance(module.attn1, model_misc.attn_class):
                hook = module.attn1.register_forward_hook(collect_compression_influence_hook, with_kwargs=True)
                all_hooks.append(hook)
            if isinstance(module.ff, DiTFastAttnFFN):
                hook = module.ff.register_forward_hook(collect_compression_influence_hook, with_kwargs=True)
                all_hooks.append(hook)

    for i, (args, kwargs) in enumerate(dataloader()):
        print(f">>> compression influence sample {i} <<<")
        model_misc.dit_inference_with_timeinfo(pipe, *args, **kwargs)

    # remove hooks
    for hook in all_hooks:
        hook.remove()

    return layer_compression_influences


def fisher_info_planning(
    layer_fisher_info,
):
    layer_fisher_info = get_layer_fisher_info(model, calib_x, n_steps)
    breakpoint()
    1 == 1
    pass

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union
from diffusers.pipelines.dit.pipeline_dit import DiTPipeline, randn_tensor
from diffusers.models.attention_processor import Attention, AttnProcessor2_0
from diffusers.models.attention import FeedForward, BasicTransformerBlock
from ditfastattn_api.modules.dfa_ffn import DiTFastAttnFFN

block_class = BasicTransformerBlock
attn_class = Attention
ffn_class = FeedForward


def inference_fn_with_backward(
    pipe: DiTPipeline,
    class_labels: List[int],
    guidance_scale: float = 4.0,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    num_inference_steps: int = 50,
):
    batch_size = len(class_labels)
    latent_size = pipe.transformer.config.sample_size
    latent_channels = pipe.transformer.config.in_channels

    latents = randn_tensor(
        shape=(batch_size, latent_channels, latent_size, latent_size),
        generator=generator,
        device=pipe._execution_device,
        dtype=pipe.transformer.dtype,
    )
    latent_model_input = torch.cat([latents] * 2) if guidance_scale > 1 else latents

    class_labels = torch.tensor(class_labels, device=pipe._execution_device).reshape(-1)
    class_null = torch.tensor([1000] * batch_size, device=pipe._execution_device)
    class_labels_input = torch.cat([class_labels, class_null], 0) if guidance_scale > 1 else class_labels

    # set step values
    pipe.scheduler.set_timesteps(num_inference_steps)
    for timestep_index, t in pipe.progress_bar(enumerate(pipe.scheduler.timesteps)):
        if guidance_scale > 1:
            half = latent_model_input[: len(latent_model_input) // 2]
            latent_model_input = torch.cat([half, half], dim=0)
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

        timesteps = t
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            # This would be a good case for the `match` statement (Python 3.10+)
            is_mps = latent_model_input.device.type == "mps"
            if isinstance(timesteps, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=latent_model_input.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(latent_model_input.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(latent_model_input.shape[0])
        # predict noise model_output
        for name, module in pipe.transformer.named_modules():
            module.timestep_index = timestep_index
        noise_pred = pipe.transformer(
            latent_model_input.detach(), timestep=timesteps.detach(), class_labels=class_labels_input.detach()
        ).sample

        # perform guidance
        if guidance_scale > 1:
            eps, rest = noise_pred[:, :latent_channels], noise_pred[:, latent_channels:]
            cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)

            half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
            eps = torch.cat([half_eps, half_eps], dim=0)

            noise_pred = torch.cat([eps, rest], dim=1)

        # learned sigma
        if pipe.transformer.config.out_channels // 2 == latent_channels:
            model_output, _ = torch.split(noise_pred, latent_channels, dim=1)
        else:
            model_output = noise_pred

        # compute previous image: x_t -> x_t-1
        latent_model_input = pipe.scheduler.step(model_output, t.detach(), latent_model_input.detach()).prev_sample
        # compute fisher info
        # latent_model_input.pow(2).mean().backward()
        torch.log(latent_model_input).sum().backward()
        pipe.transformer.zero_grad()


@torch.no_grad()
def inference_with_timeinfo(
    pipe: DiTPipeline,
    class_labels: List[int],
    guidance_scale: float = 4.0,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    num_inference_steps: int = 50,
):
    batch_size = len(class_labels)
    latent_size = pipe.transformer.config.sample_size
    latent_channels = pipe.transformer.config.in_channels

    latents = randn_tensor(
        shape=(batch_size, latent_channels, latent_size, latent_size),
        generator=generator,
        device=pipe._execution_device,
        dtype=pipe.transformer.dtype,
    )
    latent_model_input = torch.cat([latents] * 2) if guidance_scale > 1 else latents

    class_labels = torch.tensor(class_labels, device=pipe._execution_device).reshape(-1)
    class_null = torch.tensor([1000] * batch_size, device=pipe._execution_device)
    class_labels_input = torch.cat([class_labels, class_null], 0) if guidance_scale > 1 else class_labels

    # set step values
    pipe.scheduler.set_timesteps(num_inference_steps)
    for timestep_index, t in pipe.progress_bar(enumerate(pipe.scheduler.timesteps)):
        if guidance_scale > 1:
            half = latent_model_input[: len(latent_model_input) // 2]
            latent_model_input = torch.cat([half, half], dim=0)
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

        timesteps = t
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            # This would be a good case for the `match` statement (Python 3.10+)
            is_mps = latent_model_input.device.type == "mps"
            if isinstance(timesteps, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=latent_model_input.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(latent_model_input.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(latent_model_input.shape[0])
        # predict noise model_output
        for name, module in pipe.transformer.named_modules():
            module.timestep_index = timestep_index
            if hasattr(module, "stepi"):
                module.stepi = timestep_index
            if isinstance(module, attn_class):
                module.processor.stepi = timestep_index

        noise_pred = pipe.transformer(latent_model_input, timestep=timesteps, class_labels=class_labels_input).sample

        # perform guidance
        if guidance_scale > 1:
            eps, rest = noise_pred[:, :latent_channels], noise_pred[:, latent_channels:]
            cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)

            half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
            eps = torch.cat([half_eps, half_eps], dim=0)

            noise_pred = torch.cat([eps, rest], dim=1)

        # learned sigma
        if pipe.transformer.config.out_channels // 2 == latent_channels:
            model_output, _ = torch.split(noise_pred, latent_channels, dim=1)
        else:
            model_output = noise_pred

        # compute previous image: x_t -> x_t-1
        latent_model_input = pipe.scheduler.step(model_output, t, latent_model_input).prev_sample

            

def inference_fn_with_backward_plan_update(
    pipe: DiTPipeline,
    dfa_config,
    alpha, 
    class_labels: List[int],
    guidance_scale: float = 4.0,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    num_inference_steps: int = 50,
):
    batch_size = len(class_labels)
    latent_size = pipe.transformer.config.sample_size
    latent_channels = pipe.transformer.config.in_channels

    latents = randn_tensor(
        shape=(batch_size, latent_channels, latent_size, latent_size),
        generator=generator,
        device=pipe._execution_device,
        dtype=pipe.transformer.dtype,
    )
    latent_model_input = torch.cat([latents] * 2) if guidance_scale > 1 else latents

    class_labels = torch.tensor(class_labels, device=pipe._execution_device).reshape(-1)
    class_null = torch.tensor([1000] * batch_size, device=pipe._execution_device)
    class_labels_input = torch.cat([class_labels, class_null], 0) if guidance_scale > 1 else class_labels

    # set step values
    pipe.scheduler.set_timesteps(num_inference_steps)
    for timestep_index, t in pipe.progress_bar(enumerate(pipe.scheduler.timesteps)):
        if guidance_scale > 1:
            half = latent_model_input[: len(latent_model_input) // 2]
            latent_model_input = torch.cat([half, half], dim=0)
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

        timesteps = t
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            # This would be a good case for the `match` statement (Python 3.10+)
            is_mps = latent_model_input.device.type == "mps"
            if isinstance(timesteps, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=latent_model_input.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(latent_model_input.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(latent_model_input.shape[0])
        # predict noise model_output
        for name, module in pipe.transformer.named_modules():
            module.timestep_index = timestep_index
            if hasattr(module, "stepi"):
                module.stepi = timestep_index
            if isinstance(module, attn_class):
                module.processor.stepi = timestep_index
        noise_pred = pipe.transformer(
            latent_model_input.detach(), timestep=timesteps.detach(), class_labels=class_labels_input.detach()
        ).sample

        # perform guidance
        if guidance_scale > 1:
            eps, rest = noise_pred[:, :latent_channels], noise_pred[:, latent_channels:]
            cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)

            half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
            eps = torch.cat([half_eps, half_eps], dim=0)

            noise_pred = torch.cat([eps, rest], dim=1)

        # learned sigma
        if pipe.transformer.config.out_channels // 2 == latent_channels:
            model_output, _ = torch.split(noise_pred, latent_channels, dim=1)
        else:
            model_output = noise_pred

        # compute previous image: x_t -> x_t-1
        latent_model_input = pipe.scheduler.step(model_output, t.detach(), latent_model_input.detach()).prev_sample
        # compute fisher info
        latent_model_input.mean().backward()
        # torch.log(latent_model_input).sum().backward()
        pipe.transformer.zero_grad()

        # set dfa config after each timestep
        if timestep_index != 0:
            compress_methods = {}

            for name, module in pipe.transformer.named_modules():
                module.name = name
                if isinstance(module, block_class):
                    # for DiT
                    if isinstance(module.attn1, attn_class):
                        module.attn1.processor.need_cache_output = True
                        compress_methods[module.attn1.name] = "raw"
                        for candidate, influence in module.attn1.compression_influences.items():
                            print(f"layer name {module.attn1.name}, candidate {candidate}, influence {influence}, ratio {influence / latent_model_input.mean().pow(2)}, input {latent_model_input.mean()}")
                            if (influence / latent_model_input.mean().pow(2)).abs() <= alpha:
                                compress_methods[module.attn1.name] = candidate
                        if compress_methods[module.attn1.name] == "raw":
                            # update cached output
                            # update output residual
                            module.attn1.processor.cached_output = module.attn1.processor.cached_current_output
                        del module.attn1.processor.cached_current_output

                        if hasattr(module.attn1.processor, "cached_input"):
                            del module.attn1.processor.cached_input
                        if hasattr(module.attn1.processor, "cached_kwargs"):
                            del module.attn1.processor.cached_kwargs

                    if isinstance(module.ff, DiTFastAttnFFN):
                        module.ff.need_cache_output = True
                        compress_methods[module.ff.name] = "raw"
                        for candidate, influence in module.ff.compression_influences.items():
                            print(f"layer name {module.ff.name}, candidate {candidate}, influence {influence}, ratio {influence / latent_model_input.mean().pow(2)}")
                            if (influence / latent_model_input.mean().pow(2)).abs() <= alpha:
                                compress_methods[module.ff.name] = candidate
                        if compress_methods[module.ff.name] == "raw":
                            # update cached output
                            # update output residual
                            module.ff.cache_output = module.ff.cache_current_output
                        del module.ff.cache_current_output

                        if hasattr(module.ff, "cache_input"):
                            del module.ff.cache_input
                        if hasattr(module.ff, "cache_kwargs"):
                            del module.ff.cache_kwargs

            for layer_name, candidate in compress_methods.items():
                dfa_config.set_layer_step_method(layer_name, timestep_index, candidate)

    


def inference_fn_with_backward_plan_update_binary(
    pipe: DiTPipeline,
    dfa_config,
    alpha, 
    class_labels: List[int],
    guidance_scale: float = 4.0,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    num_inference_steps: int = 50,
):
    batch_size = len(class_labels)
    latent_size = pipe.transformer.config.sample_size
    latent_channels = pipe.transformer.config.in_channels

    latents = randn_tensor(
        shape=(batch_size, latent_channels, latent_size, latent_size),
        generator=generator,
        device=pipe._execution_device,
        dtype=pipe.transformer.dtype,
    )
    latent_model_input = torch.cat([latents] * 2) if guidance_scale > 1 else latents

    class_labels = torch.tensor(class_labels, device=pipe._execution_device).reshape(-1)
    class_null = torch.tensor([1000] * batch_size, device=pipe._execution_device)
    class_labels_input = torch.cat([class_labels, class_null], 0) if guidance_scale > 1 else class_labels

    # define hook
    # forward hook: cache input, kwarg and output to caculate influence in backward step
    def collect_in_out_forward_hook(m, input, kwargs, output):
        if isinstance(m, DiTFastAttnFFN):
            m.cache_current_output = output
            m.cache_input = input
            m.cache_kwargs = kwargs
        elif isinstance(m, attn_class):
            m.processor.cached_current_output = output
            m.processor.cached_input = input
            m.processor.cached_kwargs = kwargs

    def collect_influence_backward_hook(m, grad_input, grad_output):
        if m.timestep_index != 0:
            if not hasattr(m,"compression_influences"):
                m.compression_influences = {}
            candidates = dfa_config.get_available_candidates(m.name)
            if isinstance(m, DiTFastAttnFFN):
                output = m.cache_current_output
                input = m.cache_input
                kwargs = m.cache_kwargs
            elif isinstance(m, attn_class):
                output = m.processor.cached_current_output
                input = m.processor.cached_input
                kwargs = m.processor.cached_kwargs
            for candidate in candidates:
                dfa_config.set_layer_step_method(m.name, m.timestep_index, candidate)
                with torch.no_grad():
                    new_output = m.forward(*input, **kwargs).float()
                # breakpoint()
                influence = ((new_output - output.float()) * grad_output[0].detach()).sum().abs()
                # breakpoint()
                if candidate not in m.compression_influences:
                    m.compression_influences[candidate] = 0
                m.compression_influences[candidate] += influence
                # print(f"m name {m.name}, candidate {candidate}, influence: {influence}")
                dfa_config.set_layer_step_method(m.name, m.timestep_index, "raw")



    # set step values
    pipe.scheduler.set_timesteps(num_inference_steps)
    for timestep_index, t in pipe.progress_bar(enumerate(pipe.scheduler.timesteps)):
        # set hook, set need_cache_output to false
        all_hooks = []
        for name, module in pipe.transformer.named_modules():
            module.name = name
            if isinstance(module, block_class):
                # for DiT
                if isinstance(module.attn1, attn_class):
                    module.attn1.processor.compression_influences = {}
                    hook = module.attn1.register_forward_hook(collect_in_out_forward_hook, with_kwargs=True)
                    all_hooks.append(hook)
                    hook = module.attn1.register_full_backward_hook(collect_influence_backward_hook)
                    all_hooks.append(hook)
                    module.attn1.processor.need_cache_output = False
                    module.attn1.processor.cache_residual_forced = False
                if isinstance(module.ff, DiTFastAttnFFN):
                    module.ff.compression_influences = {}
                    hook = module.ff.register_forward_hook(collect_in_out_forward_hook, with_kwargs=True)
                    all_hooks.append(hook)
                    hook = module.ff.register_full_backward_hook(collect_influence_backward_hook)
                    all_hooks.append(hook)
                    module.ff.need_cache_output = False

        if guidance_scale > 1:
            half = latent_model_input[: len(latent_model_input) // 2]
            latent_model_input = torch.cat([half, half], dim=0)
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

        timesteps = t
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            # This would be a good case for the `match` statement (Python 3.10+)
            is_mps = latent_model_input.device.type == "mps"
            if isinstance(timesteps, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=latent_model_input.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(latent_model_input.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(latent_model_input.shape[0])
        # predict noise model_output
        for name, module in pipe.transformer.named_modules():
            module.timestep_index = timestep_index
            if hasattr(module, "stepi"):
                module.stepi = timestep_index
            if isinstance(module, attn_class):
                module.processor.stepi = timestep_index
        noise_pred = pipe.transformer(
            latent_model_input.detach(), timestep=timesteps.detach(), class_labels=class_labels_input.detach()
        ).sample

        # perform guidance
        if guidance_scale > 1:
            eps, rest = noise_pred[:, :latent_channels], noise_pred[:, latent_channels:]
            cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)

            half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
            eps = torch.cat([half_eps, half_eps], dim=0)

            noise_pred = torch.cat([eps, rest], dim=1)

        # learned sigma
        if pipe.transformer.config.out_channels // 2 == latent_channels:
            model_output, _ = torch.split(noise_pred, latent_channels, dim=1)
        else:
            model_output = noise_pred

        # compute previous image: x_t -> x_t-1
        _latent_model_input = pipe.scheduler.step(model_output, t.detach(), latent_model_input.detach()).prev_sample
        # compute fisher info
        _latent_model_input.mean().backward()
        # _latent_model_input.mean().backward()
        pipe.transformer.zero_grad()

        # remove hooks
        for hook in all_hooks:
            hook.remove()

        # remove all cached input and kwargs
        for name, module in pipe.transformer.named_modules():
            module.name = name
            if isinstance(module, block_class):
                # for DiT
                if isinstance(module.attn1, attn_class):
                    if hasattr(module.attn1.processor, "cached_input"):
                        del module.attn1.processor.cached_input
                    if hasattr(module.attn1.processor, "cached_kwargs"):
                        del module.attn1.processor.cached_kwargs

                if isinstance(module.ff, DiTFastAttnFFN):
                    if hasattr(module.ff, "cache_input"):
                        del module.ff.cache_input
                    if hasattr(module.ff, "cache_kwargs"):
                        del module.ff.cache_kwargs

        # set dfa config after each timestep
        if timestep_index != 0:
            influences = {}
            # iterate through modules and get compression influence
            for name, module in pipe.transformer.named_modules():
                module.name = name
                if isinstance(module, block_class):
                    # for DiT
                    if isinstance(module.attn1, attn_class):
                        if module.attn1.name not in influences:
                            influences[module.attn1.name] = {}
                        for candidate, cached_influence in module.attn1.compression_influences.items():
                            if candidate not in influences[module.attn1.name]:
                                influences[module.attn1.name][candidate] = 0
                            influences[module.attn1.name][candidate] += cached_influence
                        

                    if isinstance(module.ff, DiTFastAttnFFN):
                        if module.ff.name not in influences:
                            influences[module.ff.name] = {}
                        for candidate, cached_influence in module.ff.compression_influences.items():
                            if candidate not in influences[module.ff.name]:
                                influences[module.ff.name][candidate] = 0
                            influences[module.ff.name][candidate] += cached_influence
            # sort the influence
            sorted_layer_compression_influences = []
            for layer_name, layer_influences in influences.items():
                for candidate, influence in layer_influences.items():
                    sorted_layer_compression_influences.append((layer_name, candidate, influence))
            sorted_layer_compression_influences.sort(key=lambda x: x[2])
            print(sorted_layer_compression_influences)
            # conduct binary search
            l_idx = 0
            r_idx = len(sorted_layer_compression_influences)
            m_idx = (r_idx - l_idx) // 2
            tol = 1e-3
            l = 1
            # breakpoint()
            while r_idx - l_idx >= 2:
                # choose compression method that achieve highest compression ratio
                # print(dfa_config)
                # reset method of all layers to raw
                dfa_config.reset_step_method(timestep_index)
                for i in range(m_idx):
                    dfa_config.set_layer_step_method(sorted_layer_compression_influences[i][0], timestep_index, sorted_layer_compression_influences[i][1])
                # dfa_config.display_step_method(timestep_index)
                # breakpoint()
                with torch.no_grad():
                    temp_noise_pred = pipe.transformer(
                        latent_model_input.detach(), timestep=timesteps.detach(), class_labels=class_labels_input.detach()
                        ).sample
                    # perform guidance
                    if guidance_scale > 1:
                        eps, rest = temp_noise_pred[:, :latent_channels], temp_noise_pred[:, latent_channels:]
                        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)

                        half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
                        eps = torch.cat([half_eps, half_eps], dim=0)

                        temp_noise_pred = torch.cat([eps, rest], dim=1)
                    # learned sigma
                    if pipe.transformer.config.out_channels // 2 == latent_channels:
                        temp_model_output, _ = torch.split(temp_noise_pred, latent_channels, dim=1)
                    else:
                        temp_model_output = temp_noise_pred
                    temp_latent_model_input = pipe.scheduler.step(temp_model_output, t.detach(), latent_model_input.detach()).prev_sample
                # print(torch.max(noise_pred, temp_noise_pred) + 1e-6)
                # print(noise_pred.sum())
                # print(temp_noise_pred.sum())         
                diff = (_latent_model_input - temp_latent_model_input) / (torch.max(_latent_model_input, temp_latent_model_input) + 1e-6)
                # print(f"diff {diff}")
                l = diff.abs().clip(0,10).mean()
                # add up influence
                influence = 0
                for name, module in pipe.transformer.named_modules():
                    module.name = name
                    if isinstance(module, block_class):
                        # for DiT
                        if isinstance(module.attn1, attn_class):
                            current_method = dfa_config.layers[module.attn1.name]["kwargs"]['steps_method'][timestep_index]
                            if current_method != "raw":
                                influence = influence + influences[module.attn1.name][current_method]
                        if isinstance(module.ff, DiTFastAttnFFN):
                            current_method = dfa_config.layers[module.ff.name]["kwargs"]['steps_method'][timestep_index]
                            if current_method != "raw":
                                influence = influence + influences[module.ff.name][current_method]
                print(f"l idx {l_idx}, m idx {m_idx}, r idx {r_idx}, l {l:,.3f}, influence ratio {(influence / latent_model_input.mean().pow(2))}, alpha {alpha:,.3f}")
                if l <= alpha:
                    l_idx = m_idx
                elif l > alpha:
                    r_idx = m_idx
                m_idx = (r_idx + l_idx) // 2

                if abs(l - alpha) < tol:
                    break

            if l_idx == 0:
                dfa_config.reset_step_method(timestep_index)
     

        for name, module in pipe.transformer.named_modules():
            module.name = name
            if isinstance(module, block_class):
                # for DiT
                if isinstance(module.attn1, attn_class):
                    module.attn1.processor.need_cache_output = True
                    if module.attn1.processor.steps_method[timestep_index] == "raw":
                        module.attn1.processor.cache_residual_forced = True
                if isinstance(module.ff, DiTFastAttnFFN):
                    module.ff.need_cache_output = True

        # rerun the model to get correct output 
        noise_pred = pipe.transformer(
            latent_model_input.detach(), timestep=timesteps.detach(), class_labels=class_labels_input.detach()
        ).sample

        # perform guidance
        if guidance_scale > 1:
            eps, rest = noise_pred[:, :latent_channels], noise_pred[:, latent_channels:]
            cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)

            half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
            eps = torch.cat([half_eps, half_eps], dim=0)

            noise_pred = torch.cat([eps, rest], dim=1)

        # learned sigma
        if pipe.transformer.config.out_channels // 2 == latent_channels:
            model_output, _ = torch.split(noise_pred, latent_channels, dim=1)
        else:
            model_output = noise_pred

        # compute previous image: x_t -> x_t-1
        latent_model_input = pipe.scheduler.step(model_output, t.detach(), latent_model_input.detach()).prev_sample

        breakpoint()
    
    for name, module in pipe.transformer.named_modules():
        module.name = name
        if isinstance(module, block_class):
            # for DiT
            if isinstance(module.attn1, attn_class):
                if hasattr(module.attn1, "cached_residual"):
                    del module.attn1.cached_residual
                module.attn1.processor.cache_residual_forced = False



def inference_fn_with_backward_metric_check(
    pipe: DiTPipeline,
    dfa_config,
    alpha, 
    class_labels: List[int],
    guidance_scale: float = 4.0,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    num_inference_steps: int = 50,
):
    batch_size = len(class_labels)
    latent_size = pipe.transformer.config.sample_size
    latent_channels = pipe.transformer.config.in_channels

    latents = randn_tensor(
        shape=(batch_size, latent_channels, latent_size, latent_size),
        generator=generator,
        device=pipe._execution_device,
        dtype=pipe.transformer.dtype,
    )
    latent_model_input = torch.cat([latents] * 2) if guidance_scale > 1 else latents

    class_labels = torch.tensor(class_labels, device=pipe._execution_device).reshape(-1)
    class_null = torch.tensor([1000] * batch_size, device=pipe._execution_device)
    class_labels_input = torch.cat([class_labels, class_null], 0) if guidance_scale > 1 else class_labels

    # define hook
    # forward hook: cache input, kwarg and output to caculate influence in backward step
    def collect_in_out_forward_hook(m, input, kwargs, output):
        if isinstance(m, DiTFastAttnFFN):
            m.cache_current_output = output
            m.cache_input = input
            m.cache_kwargs = kwargs
        elif isinstance(m, attn_class):
            m.processor.cached_current_output = output
            m.processor.cached_input = input
            m.processor.cached_kwargs = kwargs

    def collect_influence_backward_hook(m, grad_input, grad_output):
        if m.timestep_index != 0:
            if not hasattr(m,"compression_influences"):
                m.compression_influences = {}
            candidates = dfa_config.get_available_candidates(m.name)
            if isinstance(m, DiTFastAttnFFN):
                output = m.cache_current_output
                input = m.cache_input
                kwargs = m.cache_kwargs
            elif isinstance(m, attn_class):
                output = m.processor.cached_current_output
                input = m.processor.cached_input
                kwargs = m.processor.cached_kwargs
            for candidate in candidates:
                dfa_config.set_layer_step_method(m.name, m.timestep_index, candidate)
                with torch.no_grad():
                    new_output = m.forward(*input, **kwargs).float()
                # breakpoint()
                grad_value = grad_output[0].detach()
                influence = ((new_output - output.float()) * grad_value).sum().detach()
                # breakpoint()
                if candidate not in m.compression_influences:
                    m.compression_influences[candidate] = 0
                m.compression_influences[candidate] += influence
                # print(f"m name {m.name}, candidate {candidate}, influence: {influence}")
                dfa_config.set_layer_step_method(m.name, m.timestep_index, "raw")



    # set step values
    pipe.scheduler.set_timesteps(num_inference_steps)
    for timestep_index, t in pipe.progress_bar(enumerate(pipe.scheduler.timesteps)):
        # set hook, set need_cache_output to false
        all_hooks = []
        for name, module in pipe.transformer.named_modules():
            module.name = name
            if isinstance(module, block_class):
                # for DiT
                if isinstance(module.attn1, attn_class):
                    module.attn1.processor.compression_influences = {}
                    hook = module.attn1.register_forward_hook(collect_in_out_forward_hook, with_kwargs=True)
                    all_hooks.append(hook)
                    hook = module.attn1.register_full_backward_hook(collect_influence_backward_hook)
                    all_hooks.append(hook)
                    module.attn1.processor.need_cache_output = False
                    module.attn1.processor.cache_residual_forced = False
                if isinstance(module.ff, DiTFastAttnFFN):
                    module.ff.compression_influences = {}
                    hook = module.ff.register_forward_hook(collect_in_out_forward_hook, with_kwargs=True)
                    all_hooks.append(hook)
                    hook = module.ff.register_full_backward_hook(collect_influence_backward_hook)
                    all_hooks.append(hook)
                    module.ff.need_cache_output = False

        if guidance_scale > 1:
            half = latent_model_input[: len(latent_model_input) // 2]
            latent_model_input = torch.cat([half, half], dim=0)
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

        timesteps = t
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            # This would be a good case for the `match` statement (Python 3.10+)
            is_mps = latent_model_input.device.type == "mps"
            if isinstance(timesteps, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=latent_model_input.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(latent_model_input.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(latent_model_input.shape[0])
        # predict noise model_output
        for name, module in pipe.transformer.named_modules():
            module.timestep_index = timestep_index
            if hasattr(module, "stepi"):
                module.stepi = timestep_index
            if isinstance(module, attn_class):
                module.processor.stepi = timestep_index
        noise_pred = pipe.transformer(
            latent_model_input.detach(), timestep=timesteps.detach(), class_labels=class_labels_input.detach()
        ).sample

        # perform guidance
        if guidance_scale > 1:
            eps, rest = noise_pred[:, :latent_channels], noise_pred[:, latent_channels:]
            cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)

            half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
            eps = torch.cat([half_eps, half_eps], dim=0)

            noise_pred = torch.cat([eps, rest], dim=1)

        # learned sigma
        if pipe.transformer.config.out_channels // 2 == latent_channels:
            model_output, _ = torch.split(noise_pred, latent_channels, dim=1)
        else:
            model_output = noise_pred

        # compute previous image: x_t -> x_t-1
        _latent_model_input = pipe.scheduler.step(model_output, t.detach(), latent_model_input.detach()).prev_sample
        # compute fisher info
        _latent_model_input.mean().backward()
        # _latent_model_input.mean().backward()
        pipe.transformer.zero_grad()

        # remove hooks
        for hook in all_hooks:
            hook.remove()

        # remove all cached input and kwargs
        for name, module in pipe.transformer.named_modules():
            module.name = name
            if isinstance(module, block_class):
                # for DiT
                if isinstance(module.attn1, attn_class):
                    
                    if hasattr(module.attn1.processor, "cached_input"):
                        module.attn1.processor.need_cache_output = False
                        del module.attn1.processor.cached_input
                    if hasattr(module.attn1.processor, "cached_kwargs"):
                        module.ff.need_cache_output = False
                        del module.attn1.processor.cached_kwargs

                if isinstance(module.ff, DiTFastAttnFFN):
                    if hasattr(module.ff, "cache_input"):
                        del module.ff.cache_input
                    if hasattr(module.ff, "cache_kwargs"):
                        del module.ff.cache_kwargs

        # set dfa config after each timestep
        if timestep_index != 0:
            influences = {}
            # iterate through modules and get compression influence
            for name, module in pipe.transformer.named_modules():
                module.name = name
                if isinstance(module, block_class):
                    # for DiT
                    if isinstance(module.attn1, attn_class):
                        if module.attn1.name not in influences:
                            influences[module.attn1.name] = {}
                        for candidate, cached_influence in module.attn1.compression_influences.items():
                            if candidate not in influences[module.attn1.name]:
                                influences[module.attn1.name][candidate] = [0,0]
                            influences[module.attn1.name][candidate][0] += abs(cached_influence)
                            # print("before: ===============================")
                            # dfa_config.display_step_method(timestep_index)
                            # print("=======================================")
                            dfa_config.set_layer_step_method(module.attn1.name, timestep_index, candidate)
                            # print("after: ===============================")
                            # dfa_config.display_step_method(timestep_index)
                            # print("======================================")
                            # breakpoint()
                            with torch.no_grad():
                                temp_noise_pred = pipe.transformer(
                                    latent_model_input.detach(), timestep=timesteps.detach(), class_labels=class_labels_input.detach()
                                    ).sample
                                # perform guidance
                                if guidance_scale > 1:
                                    eps, rest = temp_noise_pred[:, :latent_channels], noise_pred[:, latent_channels:]
                                    cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)

                                    half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
                                    eps = torch.cat([half_eps, half_eps], dim=0)

                                    temp_noise_pred = torch.cat([eps, rest], dim=1)

                                    # learned sigma
                                    if pipe.transformer.config.out_channels // 2 == latent_channels:
                                        temp_model_output, _ = torch.split(temp_noise_pred, latent_channels, dim=1)
                                    else:
                                        temp_model_output = temp_noise_pred
                                    temp_latent_model_input = pipe.scheduler.step(temp_model_output, t.detach(), latent_model_input.detach()).prev_sample
                            # print(torch.max(noise_pred, temp_noise_pred) + 1e-6)
                            # print(noise_pred.sum())
                            # print(temp_noise_pred.sum())         
                            diff = (noise_pred - temp_noise_pred) / (torch.max(noise_pred, temp_noise_pred) + 1e-6)
                            # print(f"diff {diff}")
                            l = diff.abs().clip(0,10).mean()
                            influences[module.attn1.name][candidate][1] += abs(_latent_model_input.mean() - temp_latent_model_input.mean()).detach()
                            print(f"name {module.attn1.name}, method {candidate}\ninfluence: {influences[module.attn1.name][candidate]} \n \
                                  l: {(_latent_model_input.mean() - temp_latent_model_input.mean())}\n \
                                  O: {_latent_model_input.mean()}\n \
                                  O_candidate: {temp_latent_model_input.mean()}\n \
                                  noise_pred: {noise_pred.mean()}\n \
                                  temp_noise_pred: {temp_noise_pred.mean()}\n \
                                  ratio {influences[module.attn1.name][candidate][0] / (_latent_model_input.mean() - temp_latent_model_input.mean())}")
                            dfa_config.set_layer_step_method(module.attn1.name, timestep_index, "raw")
                        

                    if isinstance(module.ff, DiTFastAttnFFN):
                        if module.ff.name not in influences:
                            influences[module.ff.name] = {}
                        for candidate, cached_influence in module.ff.compression_influences.items():
                            if candidate not in influences[module.ff.name]:
                                influences[module.ff.name][candidate] = [0,0]
                            influences[module.ff.name][candidate][0] += abs(cached_influence)

                            dfa_config.set_layer_step_method(module.ff.name, timestep_index, candidate)
                            # dfa_config.display_step_method(timestep_index)
                            # breakpoint()
                            with torch.no_grad():
                                temp_noise_pred = pipe.transformer(
                                    latent_model_input.detach(), timestep=timesteps.detach(), class_labels=class_labels_input.detach()
                                    ).sample
                                # perform guidance
                                if guidance_scale > 1:
                                    eps, rest = temp_noise_pred[:, :latent_channels], noise_pred[:, latent_channels:]
                                    cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)

                                    half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
                                    eps = torch.cat([half_eps, half_eps], dim=0)

                                    temp_noise_pred = torch.cat([eps, rest], dim=1)

                                    # learned sigma
                                    if pipe.transformer.config.out_channels // 2 == latent_channels:
                                        temp_model_output, _ = torch.split(temp_noise_pred, latent_channels, dim=1)
                                    else:
                                        temp_model_output = temp_noise_pred
                                    temp_latent_model_input = pipe.scheduler.step(temp_model_output, t.detach(), latent_model_input.detach()).prev_sample
                            influences[module.ff.name][candidate][1] += abs(_latent_model_input.mean() - temp_latent_model_input.mean()).detach()
                            # print(torch.max(noise_pred, temp_noise_pred) + 1e-6)
                            # print(noise_pred.sum())
                            # print(temp_noise_pred.sum())         
                            diff = (noise_pred - temp_noise_pred) / (torch.max(noise_pred, temp_noise_pred) + 1e-6)
                            # print(f"diff {diff}")
                            l = diff.abs().clip(0,10).mean()
                            print(f"name {module.ff.name}, method {candidate}\ninfluence: {influences[module.ff.name][candidate][0] / _latent_model_input.pow(2).mean()} \n l: {(_latent_model_input.mean() - temp_latent_model_input.mean())}")
                            dfa_config.set_layer_step_method(module.ff.name, timestep_index, "raw")

            # sort the influence
            sorted_layer_compression_influences = []
            for layer_name, layer_influences in influences.items():
                for candidate, influence in layer_influences.items():
                    sorted_layer_compression_influences.append((layer_name, candidate, influence[0], influence[1]))
            sorted_layer_compression_influences.sort(key=lambda x: x[2])
            print("sort by influence")
            print(f"{sorted_layer_compression_influences}")
            sorted_layer_compression_influences.sort(key=lambda x: x[3])
            print("sort by loss")
            print(f"{sorted_layer_compression_influences}")

        for name, module in pipe.transformer.named_modules():
            module.name = name
            if isinstance(module, block_class):
                # for DiT
                if isinstance(module.attn1, attn_class):
                    module.attn1.processor.need_cache_output = True
                    if module.attn1.processor.steps_method[timestep_index] == "raw":
                        module.attn1.processor.cache_residual_forced = True
                if isinstance(module.ff, DiTFastAttnFFN):
                    module.ff.need_cache_output = True

        # rerun the model to get correct output 
        noise_pred = pipe.transformer(
            latent_model_input.detach(), timestep=timesteps.detach(), class_labels=class_labels_input.detach()
        ).sample

        # perform guidance
        if guidance_scale > 1:
            eps, rest = noise_pred[:, :latent_channels], noise_pred[:, latent_channels:]
            cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)

            half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
            eps = torch.cat([half_eps, half_eps], dim=0)

            noise_pred = torch.cat([eps, rest], dim=1)

        # learned sigma
        if pipe.transformer.config.out_channels // 2 == latent_channels:
            model_output, _ = torch.split(noise_pred, latent_channels, dim=1)
        else:
            model_output = noise_pred

        # compute previous image: x_t -> x_t-1
        latent_model_input = pipe.scheduler.step(model_output, t.detach(), latent_model_input.detach()).prev_sample
    
    for name, module in pipe.transformer.named_modules():
            module.name = name
            if isinstance(module, block_class):
                # for DiT
                if isinstance(module.attn1, attn_class):
                    if hasattr(module.attn1, "cached_residual"):
                        del module.attn1.cached_residual
                    if module.attn1.processor.steps_method[timestep_index] == "raw":
                        module.attn1.processor.cache_residual_forced = False



def inference_fn_with_backward_plan_update_binary_advanced(
    pipe: DiTPipeline,
    dfa_config,
    alpha, 
    class_labels: List[int],
    guidance_scale: float = 4.0,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    num_inference_steps: int = 50,
):
    batch_size = len(class_labels)
    latent_size = pipe.transformer.config.sample_size
    latent_channels = pipe.transformer.config.in_channels

    latents = randn_tensor(
        shape=(batch_size, latent_channels, latent_size, latent_size),
        generator=generator,
        device=pipe._execution_device,
        dtype=pipe.transformer.dtype,
    )
    latent_model_input = torch.cat([latents] * 2) if guidance_scale > 1 else latents

    class_labels = torch.tensor(class_labels, device=pipe._execution_device).reshape(-1)
    class_null = torch.tensor([1000] * batch_size, device=pipe._execution_device)
    class_labels_input = torch.cat([class_labels, class_null], 0) if guidance_scale > 1 else class_labels

    # define hook
    # forward hook: cache input, kwarg and output to caculate influence in backward step
    def collect_in_out_forward_hook(m, input, kwargs, output):
        if isinstance(m, DiTFastAttnFFN):
            m.cache_current_output = output
            m.cache_input = input
            m.cache_kwargs = kwargs
        elif isinstance(m, attn_class):
            m.processor.cached_current_output = output
            m.processor.cached_input = input
            m.processor.cached_kwargs = kwargs

    def collect_influence_backward_hook(m, grad_input, grad_output):
        if m.timestep_index != 0:
            if not hasattr(m,"compression_influences"):
                m.compression_influences = {}
            candidates = dfa_config.get_available_candidates(m.name)
            if isinstance(m, DiTFastAttnFFN):
                output = m.cache_current_output
                input = m.cache_input
                kwargs = m.cache_kwargs
            elif isinstance(m, attn_class):
                output = m.processor.cached_current_output
                input = m.processor.cached_input
                kwargs = m.processor.cached_kwargs
            for candidate in candidates:
                dfa_config.set_layer_step_method(m.name, m.timestep_index, candidate)
                with torch.no_grad():
                    new_output = m.forward(*input, **kwargs).float()
                # breakpoint()
                influence = ((new_output - output.float()) * grad_output[0].detach()).sum().abs()
                # breakpoint()
                if candidate not in m.compression_influences:
                    m.compression_influences[candidate] = 0
                m.compression_influences[candidate] += influence
                # print(f"m name {m.name}, candidate {candidate}, influence: {influence}")
                dfa_config.set_layer_step_method(m.name, m.timestep_index, "raw")



    # set step values
    pipe.scheduler.set_timesteps(num_inference_steps)
    for timestep_index, t in pipe.progress_bar(enumerate(pipe.scheduler.timesteps)):
        # set hook, set need_cache_output to false
        all_hooks = []
        for name, module in pipe.transformer.named_modules():
            module.name = name
            if isinstance(module, block_class):
                # for DiT
                if isinstance(module.attn1, attn_class):
                    module.attn1.processor.compression_influences = {}
                    hook = module.attn1.register_forward_hook(collect_in_out_forward_hook, with_kwargs=True)
                    all_hooks.append(hook)
                    hook = module.attn1.register_full_backward_hook(collect_influence_backward_hook)
                    all_hooks.append(hook)
                    module.attn1.processor.need_cache_output = False
                    module.attn1.processor.cache_residual_forced = False
                if isinstance(module.ff, DiTFastAttnFFN):
                    module.ff.compression_influences = {}
                    hook = module.ff.register_forward_hook(collect_in_out_forward_hook, with_kwargs=True)
                    all_hooks.append(hook)
                    hook = module.ff.register_full_backward_hook(collect_influence_backward_hook)
                    all_hooks.append(hook)
                    module.ff.need_cache_output = False

        if guidance_scale > 1:
            half = latent_model_input[: len(latent_model_input) // 2]
            latent_model_input = torch.cat([half, half], dim=0)
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

        timesteps = t
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            # This would be a good case for the `match` statement (Python 3.10+)
            is_mps = latent_model_input.device.type == "mps"
            if isinstance(timesteps, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=latent_model_input.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(latent_model_input.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(latent_model_input.shape[0])
        # predict noise model_output
        for name, module in pipe.transformer.named_modules():
            module.timestep_index = timestep_index
            if hasattr(module, "stepi"):
                module.stepi = timestep_index
            if isinstance(module, attn_class):
                module.processor.stepi = timestep_index
        noise_pred = pipe.transformer(
            latent_model_input.detach(), timestep=timesteps.detach(), class_labels=class_labels_input.detach()
        ).sample

        # perform guidance
        if guidance_scale > 1:
            eps, rest = noise_pred[:, :latent_channels], noise_pred[:, latent_channels:]
            cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)

            half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
            eps = torch.cat([half_eps, half_eps], dim=0)

            noise_pred = torch.cat([eps, rest], dim=1)

        # learned sigma
        if pipe.transformer.config.out_channels // 2 == latent_channels:
            model_output, _ = torch.split(noise_pred, latent_channels, dim=1)
        else:
            model_output = noise_pred

        # compute previous image: x_t -> x_t-1
        _latent_model_input = pipe.scheduler.step(model_output, t.detach(), latent_model_input.detach()).prev_sample
        # compute fisher info
        _latent_model_input.mean().backward()
        # _latent_model_input.mean().backward()
        pipe.transformer.zero_grad()

        # remove hooks
        for hook in all_hooks:
            hook.remove()

        # remove all cached input and kwargs
        for name, module in pipe.transformer.named_modules():
            module.name = name
            if isinstance(module, block_class):
                # for DiT
                if isinstance(module.attn1, attn_class):
                    if hasattr(module.attn1.processor, "cached_input"):
                        del module.attn1.processor.cached_input
                    if hasattr(module.attn1.processor, "cached_kwargs"):
                        del module.attn1.processor.cached_kwargs

                if isinstance(module.ff, DiTFastAttnFFN):
                    if hasattr(module.ff, "cache_input"):
                        del module.ff.cache_input
                    if hasattr(module.ff, "cache_kwargs"):
                        del module.ff.cache_kwargs

        # set dfa config after each timestep
        if timestep_index != 0:
            influences = {}
            # iterate through modules and get compression influence
            for name, module in pipe.transformer.named_modules():
                module.name = name
                if isinstance(module, block_class):
                    # for DiT
                    if isinstance(module.attn1, attn_class):
                        if module.attn1.name not in influences:
                            influences[module.attn1.name] = {}
                        for candidate, cached_influence in module.attn1.compression_influences.items():
                            if candidate not in influences[module.attn1.name]:
                                influences[module.attn1.name][candidate] = 0
                            dfa_config.set_layer_step_method(module.attn1.name, timestep_index, candidate)
                            # dfa_config.display_step_method(timestep_index)
                            # breakpoint()
                            with torch.no_grad():
                                temp_noise_pred = pipe.transformer(
                                    latent_model_input.detach(), timestep=timesteps.detach(), class_labels=class_labels_input.detach()
                                    ).sample
                                # perform guidance
                                if guidance_scale > 1:
                                    eps, rest = temp_noise_pred[:, :latent_channels], noise_pred[:, latent_channels:]
                                    cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)

                                    half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
                                    eps = torch.cat([half_eps, half_eps], dim=0)

                                    temp_noise_pred = torch.cat([eps, rest], dim=1)

                                    # learned sigma
                                    if pipe.transformer.config.out_channels // 2 == latent_channels:
                                        temp_model_output, _ = torch.split(temp_noise_pred, latent_channels, dim=1)
                                    else:
                                        temp_model_output = temp_noise_pred
                                    temp_latent_model_input = pipe.scheduler.step(temp_model_output, t.detach(), latent_model_input.detach()).prev_sample
                            influences[module.attn1.name][candidate] += abs(_latent_model_input.mean() - temp_latent_model_input.mean()).detach()
                            dfa_config.set_layer_step_method(module.attn1.name, timestep_index, "raw")
                        

                    if isinstance(module.ff, DiTFastAttnFFN):
                        if module.ff.name not in influences:
                            influences[module.ff.name] = {}
                        for candidate, cached_influence in module.ff.compression_influences.items():
                            if candidate not in influences[module.ff.name]:
                                influences[module.ff.name][candidate] = 0
                            dfa_config.set_layer_step_method(module.ff.name, timestep_index, candidate)
                            # dfa_config.display_step_method(timestep_index)
                            # breakpoint()
                            with torch.no_grad():
                                temp_noise_pred = pipe.transformer(
                                    latent_model_input.detach(), timestep=timesteps.detach(), class_labels=class_labels_input.detach()
                                    ).sample
                                # perform guidance
                                if guidance_scale > 1:
                                    eps, rest = temp_noise_pred[:, :latent_channels], noise_pred[:, latent_channels:]
                                    cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)

                                    half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
                                    eps = torch.cat([half_eps, half_eps], dim=0)

                                    temp_noise_pred = torch.cat([eps, rest], dim=1)

                                    # learned sigma
                                    if pipe.transformer.config.out_channels // 2 == latent_channels:
                                        temp_model_output, _ = torch.split(temp_noise_pred, latent_channels, dim=1)
                                    else:
                                        temp_model_output = temp_noise_pred
                                    temp_latent_model_input = pipe.scheduler.step(temp_model_output, t.detach(), latent_model_input.detach()).prev_sample
                            influences[module.ff.name][candidate] += abs(_latent_model_input.mean() - temp_latent_model_input.mean()).detach()
                            dfa_config.set_layer_step_method(module.ff.name, timestep_index, "raw")
            # sort the influence
            sorted_layer_compression_influences = []
            for layer_name, layer_influences in influences.items():
                for candidate, influence in layer_influences.items():
                    sorted_layer_compression_influences.append((layer_name, candidate, influence))
            sorted_layer_compression_influences.sort(key=lambda x: x[2])
            print(sorted_layer_compression_influences)
            # conduct binary search
            l_idx = 0
            r_idx = len(sorted_layer_compression_influences)
            m_idx = (r_idx - l_idx) // 2
            tol = 1e-3
            l = 1
            # breakpoint()
            while r_idx - l_idx >= 2:
                # choose compression method that achieve highest compression ratio
                # print(dfa_config)
                # reset method of all layers to raw
                dfa_config.reset_step_method(timestep_index)
                for i in range(m_idx):
                    dfa_config.set_layer_step_method(sorted_layer_compression_influences[i][0], timestep_index, sorted_layer_compression_influences[i][1])
                # dfa_config.display_step_method(timestep_index)
                # breakpoint()
                with torch.no_grad():
                    temp_noise_pred = pipe.transformer(
                        latent_model_input.detach(), timestep=timesteps.detach(), class_labels=class_labels_input.detach()
                        ).sample
                    # perform guidance
                    if guidance_scale > 1:
                        eps, rest = temp_noise_pred[:, :latent_channels], noise_pred[:, latent_channels:]
                        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)

                        half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
                        eps = torch.cat([half_eps, half_eps], dim=0)

                        temp_noise_pred = torch.cat([eps, rest], dim=1)
                    # learned sigma
                    if pipe.transformer.config.out_channels // 2 == latent_channels:
                        temp_model_output, _ = torch.split(temp_noise_pred, latent_channels, dim=1)
                    else:
                        temp_model_output = temp_noise_pred
                    temp_latent_model_input = pipe.scheduler.step(temp_model_output, t.detach(), latent_model_input.detach()).prev_sample
                # print(torch.max(noise_pred, temp_noise_pred) + 1e-6)
                # print(noise_pred.sum())
                # print(temp_noise_pred.sum())         
                diff = (_latent_model_input - temp_latent_model_input) / (torch.max(_latent_model_input, temp_latent_model_input) + 1e-6)
                # print(f"diff {diff}")
                l = diff.abs().clip(0,10).mean()
                # add up influence
                influence = 0
                for name, module in pipe.transformer.named_modules():
                    module.name = name
                    if isinstance(module, block_class):
                        # for DiT
                        if isinstance(module.attn1, attn_class):
                            current_method = dfa_config.layers[module.attn1.name]["kwargs"]['steps_method'][timestep_index]
                            if current_method != "raw":
                                influence = influence + influences[module.attn1.name][current_method]
                        if isinstance(module.ff, DiTFastAttnFFN):
                            current_method = dfa_config.layers[module.ff.name]["kwargs"]['steps_method'][timestep_index]
                            if current_method != "raw":
                                influence = influence + influences[module.ff.name][current_method]
                print(f"l idx {l_idx}, m idx {m_idx}, r idx {r_idx}, l {l:,.3f}, influence ratio {(influence / latent_model_input.mean().pow(2))}, alpha {alpha:,.3f}")
                if l <= alpha:
                    l_idx = m_idx
                elif l > alpha:
                    r_idx = m_idx
                m_idx = (r_idx + l_idx) // 2

                if abs(l - alpha) < tol:
                    break

            if l_idx == 0:
                dfa_config.reset_step_method(timestep_index)
     

        for name, module in pipe.transformer.named_modules():
            module.name = name
            if isinstance(module, block_class):
                # for DiT
                if isinstance(module.attn1, attn_class):
                    module.attn1.processor.need_cache_output = True
                    if module.attn1.processor.steps_method[timestep_index] == "raw":
                        module.attn1.processor.cache_residual_forced = True
                if isinstance(module.ff, DiTFastAttnFFN):
                    module.ff.need_cache_output = True

        # rerun the model to get correct output 
        noise_pred = pipe.transformer(
            latent_model_input.detach(), timestep=timesteps.detach(), class_labels=class_labels_input.detach()
        ).sample

        # perform guidance
        if guidance_scale > 1:
            eps, rest = noise_pred[:, :latent_channels], noise_pred[:, latent_channels:]
            cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)

            half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
            eps = torch.cat([half_eps, half_eps], dim=0)

            noise_pred = torch.cat([eps, rest], dim=1)

        # learned sigma
        if pipe.transformer.config.out_channels // 2 == latent_channels:
            model_output, _ = torch.split(noise_pred, latent_channels, dim=1)
        else:
            model_output = noise_pred

        # compute previous image: x_t -> x_t-1
        latent_model_input = pipe.scheduler.step(model_output, t.detach(), latent_model_input.detach()).prev_sample
    
    for name, module in pipe.transformer.named_modules():
            module.name = name
            if isinstance(module, block_class):
                # for DiT
                if isinstance(module.attn1, attn_class):
                    if hasattr(module.attn1, "cached_residual"):
                        del module.attn1.cached_residual
                    if module.attn1.processor.steps_method[timestep_index] == "raw":
                        module.attn1.processor.cache_residual_forced = False

# 2 phase version: phase 1 set output share, phase 2 set window attn
def inference_fn_with_backward_plan_update_binary_two_phase(
    pipe: DiTPipeline,
    dfa_config,
    alpha, 
    class_labels: List[int],
    guidance_scale: float = 4.0,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    num_inference_steps: int = 50,
):
    batch_size = len(class_labels)
    latent_size = pipe.transformer.config.sample_size
    latent_channels = pipe.transformer.config.in_channels

    latents = randn_tensor(
        shape=(batch_size, latent_channels, latent_size, latent_size),
        generator=generator,
        device=pipe._execution_device,
        dtype=pipe.transformer.dtype,
    )
    latent_model_input = torch.cat([latents] * 2) if guidance_scale > 1 else latents

    class_labels = torch.tensor(class_labels, device=pipe._execution_device).reshape(-1)
    class_null = torch.tensor([1000] * batch_size, device=pipe._execution_device)
    class_labels_input = torch.cat([class_labels, class_null], 0) if guidance_scale > 1 else class_labels

    # define hook
    # forward hook: cache input, kwarg and output to caculate influence in backward step
    def collect_in_out_forward_hook(m, input, kwargs, output):
        if isinstance(m, DiTFastAttnFFN):
            m.cache_current_output = output
            m.cache_input = input
            m.cache_kwargs = kwargs
        elif isinstance(m, attn_class):
            m.processor.cached_current_output = output
            m.processor.cached_input = input
            m.processor.cached_kwargs = kwargs

    def collect_influence_backward_hook(m, grad_input, grad_output):
        if m.timestep_index != 0:
            if not hasattr(m,"compression_influences"):
                m.compression_influences = {}
            candidates = dfa_config.get_available_candidates(m.name)
            if isinstance(m, DiTFastAttnFFN):
                output = m.cache_current_output
                input = m.cache_input
                kwargs = m.cache_kwargs
            elif isinstance(m, attn_class):
                output = m.processor.cached_current_output
                input = m.processor.cached_input
                kwargs = m.processor.cached_kwargs
            for candidate in candidates:
                dfa_config.set_layer_step_method(m.name, m.timestep_index, candidate)
                with torch.no_grad():
                    new_output = m.forward(*input, **kwargs).float()
                # breakpoint()
                influence = ((new_output - output.float()) * grad_output[0].detach()).sum().abs()
                # breakpoint()
                if candidate not in m.compression_influences:
                    m.compression_influences[candidate] = 0
                m.compression_influences[candidate] += influence
                # print(f"m name {m.name}, candidate {candidate}, influence: {influence}")
                dfa_config.set_layer_step_method(m.name, m.timestep_index, "raw")



    # set step values
    pipe.scheduler.set_timesteps(num_inference_steps)
    for timestep_index, t in pipe.progress_bar(enumerate(pipe.scheduler.timesteps)):
        # set hook, set need_cache_output to false
        all_hooks = []
        for name, module in pipe.transformer.named_modules():
            module.name = name
            if isinstance(module, block_class):
                # for DiT
                if isinstance(module.attn1, attn_class):
                    module.attn1.processor.compression_influences = {}
                    hook = module.attn1.register_forward_hook(collect_in_out_forward_hook, with_kwargs=True)
                    all_hooks.append(hook)
                    hook = module.attn1.register_full_backward_hook(collect_influence_backward_hook)
                    all_hooks.append(hook)
                    module.attn1.processor.need_cache_output = False
                    module.attn1.processor.cache_residual_forced = False
                if isinstance(module.ff, DiTFastAttnFFN):
                    module.ff.compression_influences = {}
                    hook = module.ff.register_forward_hook(collect_in_out_forward_hook, with_kwargs=True)
                    all_hooks.append(hook)
                    hook = module.ff.register_full_backward_hook(collect_influence_backward_hook)
                    all_hooks.append(hook)
                    module.ff.need_cache_output = False

        if guidance_scale > 1:
            half = latent_model_input[: len(latent_model_input) // 2]
            latent_model_input = torch.cat([half, half], dim=0)
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

        timesteps = t
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            # This would be a good case for the `match` statement (Python 3.10+)
            is_mps = latent_model_input.device.type == "mps"
            if isinstance(timesteps, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=latent_model_input.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(latent_model_input.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(latent_model_input.shape[0])
        # predict noise model_output
        for name, module in pipe.transformer.named_modules():
            module.timestep_index = timestep_index
            if hasattr(module, "stepi"):
                module.stepi = timestep_index
            if isinstance(module, attn_class):
                module.processor.stepi = timestep_index
        noise_pred = pipe.transformer(
            latent_model_input.detach(), timestep=timesteps.detach(), class_labels=class_labels_input.detach()
        ).sample

        # perform guidance
        if guidance_scale > 1:
            eps, rest = noise_pred[:, :latent_channels], noise_pred[:, latent_channels:]
            cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)

            half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
            eps = torch.cat([half_eps, half_eps], dim=0)

            noise_pred = torch.cat([eps, rest], dim=1)

        # learned sigma
        if pipe.transformer.config.out_channels // 2 == latent_channels:
            model_output, _ = torch.split(noise_pred, latent_channels, dim=1)
        else:
            model_output = noise_pred

        # compute previous image: x_t -> x_t-1
        _latent_model_input = pipe.scheduler.step(model_output, t.detach(), latent_model_input.detach()).prev_sample
        # compute fisher info
        _latent_model_input.mean().backward()
        # _latent_model_input.mean().backward()
        pipe.transformer.zero_grad()

        # remove hooks
        for hook in all_hooks:
            hook.remove()

        # remove all cached input and kwargs
        for name, module in pipe.transformer.named_modules():
            module.name = name
            if isinstance(module, block_class):
                # for DiT
                if isinstance(module.attn1, attn_class):
                    if hasattr(module.attn1.processor, "cached_input"):
                        del module.attn1.processor.cached_input
                    if hasattr(module.attn1.processor, "cached_kwargs"):
                        del module.attn1.processor.cached_kwargs

                if isinstance(module.ff, DiTFastAttnFFN):
                    if hasattr(module.ff, "cache_input"):
                        del module.ff.cache_input
                    if hasattr(module.ff, "cache_kwargs"):
                        del module.ff.cache_kwargs

        # set dfa config after each timestep
        if timestep_index != 0:
            influences = {}
            # iterate through modules and get compression influence
            for name, module in pipe.transformer.named_modules():
                module.name = name
                if isinstance(module, block_class):
                    # for DiT
                    if isinstance(module.attn1, attn_class):
                        if module.attn1.name not in influences:
                            influences[module.attn1.name] = {}
                        for candidate, cached_influence in module.attn1.compression_influences.items():
                            if candidate not in influences[module.attn1.name]:
                                influences[module.attn1.name][candidate] = 0
                            influences[module.attn1.name][candidate] += cached_influence
                        

                    if isinstance(module.ff, DiTFastAttnFFN):
                        if module.ff.name not in influences:
                            influences[module.ff.name] = {}
                        for candidate, cached_influence in module.ff.compression_influences.items():
                            if candidate not in influences[module.ff.name]:
                                influences[module.ff.name][candidate] = 0
                            influences[module.ff.name][candidate] += cached_influence
            # sort the influence
            sorted_layer_compression_influences = []
            for layer_name, layer_influences in influences.items():
                for candidate, influence in layer_influences.items():
                    sorted_layer_compression_influences.append((layer_name, candidate, influence))
            sorted_layer_compression_influences.sort(key=lambda x: x[2])
            print(sorted_layer_compression_influences)
            # conduct binary search
            l_idx = 0
            r_idx = len(sorted_layer_compression_influences)
            m_idx = (r_idx - l_idx) // 2
            tol = 1e-3
            l = 1
            # breakpoint()
            while r_idx - l_idx >= 2:
                # choose compression method that achieve highest compression ratio
                # print(dfa_config)
                # reset method of all layers to raw
                dfa_config.reset_step_method(timestep_index)
                for i in range(m_idx):
                    dfa_config.set_layer_step_method(sorted_layer_compression_influences[i][0], timestep_index, sorted_layer_compression_influences[i][1])
                # dfa_config.display_step_method(timestep_index)
                # breakpoint()
                with torch.no_grad():
                    temp_noise_pred = pipe.transformer(
                        latent_model_input.detach(), timestep=timesteps.detach(), class_labels=class_labels_input.detach()
                        ).sample
                    # perform guidance
                    if guidance_scale > 1:
                        eps, rest = temp_noise_pred[:, :latent_channels], noise_pred[:, latent_channels:]
                        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)

                        half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
                        eps = torch.cat([half_eps, half_eps], dim=0)

                        temp_noise_pred = torch.cat([eps, rest], dim=1)
                    # learned sigma
                    if pipe.transformer.config.out_channels // 2 == latent_channels:
                        temp_model_output, _ = torch.split(temp_noise_pred, latent_channels, dim=1)
                    else:
                        temp_model_output = temp_noise_pred
                    temp_latent_model_input = pipe.scheduler.step(temp_model_output, t.detach(), latent_model_input.detach()).prev_sample
                # print(torch.max(noise_pred, temp_noise_pred) + 1e-6)
                # print(noise_pred.sum())
                # print(temp_noise_pred.sum())         
                diff = (_latent_model_input - temp_latent_model_input) / (torch.max(_latent_model_input, temp_latent_model_input) + 1e-6)
                # print(f"diff {diff}")
                l = diff.abs().clip(0,10).mean()
                # add up influence
                influence = 0
                for name, module in pipe.transformer.named_modules():
                    module.name = name
                    if isinstance(module, block_class):
                        # for DiT
                        if isinstance(module.attn1, attn_class):
                            current_method = dfa_config.layers[module.attn1.name]["kwargs"]['steps_method'][timestep_index]
                            if current_method != "raw":
                                influence = influence + influences[module.attn1.name][current_method]
                        if isinstance(module.ff, DiTFastAttnFFN):
                            current_method = dfa_config.layers[module.ff.name]["kwargs"]['steps_method'][timestep_index]
                            if current_method != "raw":
                                influence = influence + influences[module.ff.name][current_method]
                print(f"l idx {l_idx}, m idx {m_idx}, r idx {r_idx}, l {l:,.3f}, influence ratio {(influence / latent_model_input.mean().pow(2))}, alpha {alpha:,.3f}")
                if l <= alpha:
                    l_idx = m_idx
                elif l > alpha:
                    r_idx = m_idx
                m_idx = (r_idx + l_idx) // 2

                if abs(l - alpha) < tol:
                    break

            if l_idx == 0:
                dfa_config.reset_step_method(timestep_index)
     

        for name, module in pipe.transformer.named_modules():
            module.name = name
            if isinstance(module, block_class):
                # for DiT
                if isinstance(module.attn1, attn_class):
                    module.attn1.processor.need_cache_output = True
                    if module.attn1.processor.steps_method[timestep_index] == "raw":
                        module.attn1.processor.cache_residual_forced = True
                if isinstance(module.ff, DiTFastAttnFFN):
                    module.ff.need_cache_output = True

        # rerun the model to get correct output 
        noise_pred = pipe.transformer(
            latent_model_input.detach(), timestep=timesteps.detach(), class_labels=class_labels_input.detach()
        ).sample

        # perform guidance
        if guidance_scale > 1:
            eps, rest = noise_pred[:, :latent_channels], noise_pred[:, latent_channels:]
            cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)

            half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
            eps = torch.cat([half_eps, half_eps], dim=0)

            noise_pred = torch.cat([eps, rest], dim=1)

        # learned sigma
        if pipe.transformer.config.out_channels // 2 == latent_channels:
            model_output, _ = torch.split(noise_pred, latent_channels, dim=1)
        else:
            model_output = noise_pred

        # compute previous image: x_t -> x_t-1
        latent_model_input = pipe.scheduler.step(model_output, t.detach(), latent_model_input.detach()).prev_sample
    
    for name, module in pipe.transformer.named_modules():
            module.name = name
            if isinstance(module, block_class):
                # for DiT
                if isinstance(module.attn1, attn_class):
                    if hasattr(module.attn1, "cached_residual"):
                        del module.attn1.cached_residual
                    if module.attn1.processor.steps_method[timestep_index] == "raw":
                        module.attn1.processor.cache_residual_forced = False
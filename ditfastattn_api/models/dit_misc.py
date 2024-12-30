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
        latent_model_input.pow(2).mean().backward()
        # torch.log(latent_model_input).sum().backward()
        pipe.transformer.zero_grad()

        # set dfa config after each timestep
        if timestep_index != 0:
            influences = {}
            # iterate through modules and get compression influence
            for name, module in pipe.transformer.named_modules():
                module.name = name
                if isinstance(module, block_class):
                    # for DiT
                    if isinstance(module.attn1, attn_class):
                        module.attn1.processor.need_cache_output = True
                        if module.attn1.name not in influences:
                            influences[module.attn1.name] = {}
                        for candidate, cached_influence in module.attn1.compression_influences.items():
                            if candidate not in influences[module.attn1.name]:
                                influences[module.attn1.name][candidate] = 0
                            influences[module.attn1.name][candidate] += cached_influence
                        

                    if isinstance(module.ff, DiTFastAttnFFN):
                        module.ff.need_cache_output = True
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
            # conduct binary search
            l_idx = 0
            r_idx = len(sorted_layer_compression_influences)
            m_idx = (r_idx - l_idx) // 2
            tol = 1e-6
            l = 1
            while l - alpha > tol:
                # choose compression method that achieve highest compression ratio
                # print(dfa_config)
                for i in range(m_idx):
                    dfa_config.set_layer_step_method(sorted_layer_compression_influences[i][0], timestep_index, sorted_layer_compression_influences[i][1])

                temp_noise_pred = pipe.transformer(
                    latent_model_input.detach(), timestep=timesteps.detach(), class_labels=class_labels_input.detach()
                    ).sample
                
                diff = (noise_pred - temp_noise_pred) / (torch.max(noise_pred, temp_noise_pred) + 1e-6)
                l = diff.abs().clip(0,10).mean()
                print(f"l idx {l_idx}, m idx {m_idx}, r idx {r_idx}, l {l:,.3f}, influence ratio {(influence / latent_model_input.mean().pow(2)):,.3f}, alpha {alpha:,.3f}")
                if l <= alpha:
                    l_idx = m_idx
                elif l > alpha:
                    r_idx = m_idx
                m_idx = (r_idx + l_idx) // 2

                if (r_idx - l_idx) < 2 and l - alpha > tol:
                    dfa_config.reset_step_method(timestep_index)
                    break
                

            for name, module in pipe.transformer.named_modules():
                module.name = name
                if isinstance(module, block_class):
                    # for DiT
                    if isinstance(module.attn1, attn_class):
                        module.attn1.processor.need_cache_output = True
                        if dfa_config.layers[module.attn1.name]['kwargs']['steps_method'][timestep_index] == "raw":
                            # update cached output
                            # update output residual
                            module.attn1.processor.cached_output = module.attn1.processor.cached_current_output
                        if hasattr(module.attn1.processor, "cached_input"):
                            del module.attn1.processor.cached_input
                        if hasattr(module.attn1.processor, "cached_kwargs"):
                            del module.attn1.processor.cached_kwargs

                    if isinstance(module.ff, DiTFastAttnFFN):
                        module.ff.need_cache_output = True
                        if dfa_config.layers[module.ff.name]['kwargs']['steps_method'][timestep_index] == "raw":
                            # update cached output
                            # update output residual
                            module.ff.cache_output = module.ff.cache_current_output
                        del module.ff.cache_current_output
                        if hasattr(module.ff, "cache_input"):
                            del module.ff.cache_input
                        if hasattr(module.ff, "cache_kwargs"):
                            del module.ff.cache_kwargs


def inference_fn_with_output_record(
    pipe: DiTPipeline,
    class_labels: List[int],
    guidance_scale: float = 4.0,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    num_inference_steps: int = 50,
):
    output_dict = {}
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
        print(timestep_index)

        output_dict[timestep_index] = latent_model_input.detach().cpu()

    return output_dict

def inference_fn_with_metric_record(
    pipe: DiTPipeline,
    class_labels: List[int],
    guidance_scale: float = 4.0,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    num_inference_steps: int = 50,
):
    output_dict = {}
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
        print(timestep_index)

        output_dict[timestep_index] = latent_model_input.detach().cpu()

    return output_dict
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union
from diffusers.pipelines.dit.pipeline_dit import DiTPipeline, randn_tensor
from diffusers.models.attention_processor import Attention, AttnProcessor2_0
from diffusers.models.attention import FeedForward, BasicTransformerBlock

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
        latent_model_input.pow(2).mean().backward()
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

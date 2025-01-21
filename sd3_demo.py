from ditfastattn_api.api import (
    transform_model_dfa,
    dfa_test_latency,
    register_refresh_stepi_hook,
    unregister_refresh_stepi_hook,
    dfa_test_layer_latency
)
from ditfastattn_api.api import MethodSpeedup
import ditfastattn_api.models.mmdit_misc as mmdit_misc

from diffusers import DiTPipeline, AutoPipelineForText2Image, StableDiffusion3Pipeline
import torch
import random
import json
import numpy as np
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

from ditfastattn_api.fisher_info_planning import (
    update_layer_influence_new,
    update_layer_influence_two_phase,
    generate_ground_truth
)

model_misc = mmdit_misc

# model_id = "./models/stable-diffusion-3-medium-diffusers/"
model_id = "stabilityai/stable-diffusion-3-medium-diffusers"
seed = 3
n_steps = 28
resolution = 1024
calib_x = torch.randint(0, 1000, (6,), generator=torch.Generator().manual_seed(seed)).to("cuda")
pipe = StableDiffusion3Pipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
pipe = pipe.to("cuda")

with open(f"/mnt/public/hanling/mscoco/annotations/captions_val2014.json") as f:
    mscoco_anno = json.load(f)

caption_list = []
for i in calib_x:
    caption_list.append(mscoco_anno["annotations"][i]["caption"])


def flex_attn_block_mask_wrapper(q, k, v, block_mask):
    return flex_attention(q, k, v, block_mask=block_mask)
flex_attn_block_mask_compiled = torch.compile(flex_attn_block_mask_wrapper, dynamic=False)

# for layer_name in dfa_config.layer_names[:10]:
#     candidates = dfa_config.get_available_candidates(layer_name)
#     for step_i in range(1, n_steps):
#         # random select one candidate
#         choice = random.choice(candidates)
#         choice = "raw"
#         if "attn" in layer_name:
#             choice = "residual_window_attn_128"
#             choice = "output_share"
#         dfa_config.set_layer_step_method(layer_name, step_i, choice)
#         # print(f"Set {layer_name} step {step_i} to {choice}")
# dfa_config.apply_configs(verbose=True)
# results = dfa_test_latency(pipe, caption_list, num_inference_steps=n_steps, height=resolution, width=resolution)
# print("dfa latency", results)

# for layer_name in dfa_config.layer_names:
#     print(layer_name)
#     candidates = dfa_config.get_available_candidates(layer_name)
#     print(candidates)
#     for step_i in range(1, n_steps - 1):
#         # random select one candidate
#         choice = random.choice(candidates)
#         dfa_config.set_layer_step_method(layer_name, step_i, choice)
#         print(f"Set {layer_name} step {step_i} to {choice}")
# dfa_config.apply_configs(verbose=True)
# results = dfa_test_latency(pipe, caption_list, num_inference_steps=n_steps, height=512, width=512)
# print("dfa latency", results)

# plot to one figure
import matplotlib.pyplot as plt
from PIL import Image


def dataloader():
    generator = torch.Generator().manual_seed(seed)
    res = []
    calib_x = torch.randint(0, 1000, (3,), generator=generator).to("cuda")
    for i in calib_x:
        res.append(mscoco_anno["annotations"][i]["caption"])
    yield res, {"num_inference_steps": n_steps, "generator": generator}

# pipe.transformer.to(torch.float16)

latency = dfa_test_latency(pipe, caption_list, num_inference_steps=n_steps, height=resolution, width=resolution)
# dfa_config = transform_model_dfa(pipe.transformer, n_steps=n_steps*2, window_func=flex_attn_block_mask_compiled)
# generate_ground_truth(pipe, dataloader, model_misc)
dfa_config = transform_model_dfa(pipe.transformer, n_steps=n_steps, window_func=flex_attn_block_mask_compiled)

ms = MethodSpeedup()
ms.load_candidates(dfa_config.get_available_candidates(dfa_config.layer_names[0]))
latency_dict = ms.generate_headwise_latency('test', pipe, n_steps, dfa_config, caption_list, num_inference_steps=n_steps, height=resolution, width=resolution)
print(latency_dict)
breakpoint()
torch.save(latency_dict, "cache/latency_dict.json")
# latency = dfa_test_latency(pipe, caption_list, num_inference_steps=n_steps, height=resolution, width=resolution)
# breakpoint()

# width 9 subplots
fig, axs = plt.subplots(1, 9, figsize=(20, 15))  # figsize width=20

for i, threshold in enumerate(np.linspace(0, 0.00018, 9)):
# for i, threshold in enumerate([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.15, 0.175, 0.2]):
    # add
    # pipe = DiTPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")
    # dfa_config = transform_model_dfa(pipe.transformer, n_steps=n_steps)

    # reset all compression methods to "raw"
    for step in range(n_steps):
        dfa_config.reset_step_method(step)
    latency = dfa_test_latency(pipe, caption_list, num_inference_steps=n_steps, height=resolution, width=resolution)


    # add
    unregister_refresh_stepi_hook(pipe.transformer)
    # compress_methods = fisher_info_planning(layer_compression_influences, dfa_config, threshold)
    # pipe.transformer.to(torch.bfloat16)
    # update_layer_influence_new(pipe, dataloader, dfa_config, model_misc, alpha = threshold)
    # update_layer_influence_two_phase(pipe, dataloader, dfa_config, model_misc, alpha1 = min(threshold, 0.04), alpha2=max(threshold - 0.04, 0))
    print("-------start calibration--------")
    update_layer_influence_two_phase(pipe, dataloader, dfa_config, model_misc, alpha = threshold)
    # pipe.transformer.to(torch.float16)
    unregister_refresh_stepi_hook(pipe.transformer)
    register_refresh_stepi_hook(pipe.transformer, n_steps)
    # breakpoint()
    print("-------end calibration--------")
    latency = dfa_test_latency(pipe, caption_list, num_inference_steps=n_steps, height=resolution, width=resolution)
    # generate a image
    generator = torch.Generator().manual_seed(seed)
    # height stack batch images
    images = pipe(caption_list, num_inference_steps=n_steps, height=resolution, width=resolution, generator=generator).images
    width = images[0].width
    total_height = sum(img.height for img in images)
    image = Image.new("RGB", (width, total_height))
    y_offset = 0
    for img in images:
        image.paste(img, (0, y_offset))
        y_offset += img.height
    axs[i].imshow(image)
    axs[i].set_title(f"threshold {threshold:.2f} \n latency {latency:.2f}s")
    axs[i].axis("off")
    # breakpoint()
plt.savefig(f"output/{model_id.replace('/', '_')}_fisher_info_planning.png")
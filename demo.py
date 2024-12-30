from ditfastattn_api.api import (
    transform_model_dfa,
    dfa_test_latency,
    register_refresh_stepi_hook,
    unregister_refresh_stepi_hook,
)
from diffusers import DiTPipeline, AutoPipelineForText2Image
from ditfastattn_api.fisher_info_planning import (
    fisher_info_planning,
    get_layer_fisher_info,
    get_compression_method_influence,
    get_layer_influence,
    update_layer_influence_new
)
import ditfastattn_api.models.dit_misc as dit_misc
import torch
import random
import numpy as np

model_id = "facebook/DiT-XL-2-512"
seed = 9
n_steps = 20
calib_x = torch.randint(0, 1000, (6,), generator=torch.Generator().manual_seed(seed)).to("cuda")
pipe = DiTPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")

n_samples = 1
model_misc = dit_misc


def dataloader():
    generator = torch.Generator().manual_seed(seed)
    for _ in range(n_samples):
        calib_x = torch.randint(0, 1000, (6,), generator=generator).to("cuda")
        yield [calib_x], {"num_inference_steps": n_steps}


# pipe.transformer.to(torch.float32)
# layer_gradients = get_layer_fisher_info(pipe, dataloader, model_misc)
# breakpoint()

pipe.transformer.to(torch.float16)
dfa_config = transform_model_dfa(pipe.transformer, n_steps=n_steps)
latency = dfa_test_latency(pipe, calib_x, num_inference_steps=n_steps)
print("raw latency", latency)

# layer_compression_influences = get_compression_method_influence(
#     pipe, dfa_config, dataloader, layer_gradients, model_misc
# )

# pipe.transformer.to(torch.bfloat16)
# layer_compression_influences = get_layer_influence(
#     pipe, dataloader, dfa_config, model_misc
# )
# pipe.transformer.to(torch.float16)

# update_layer_influence_new(pipe, dataloader, dfa_config, model_misc)


# plot to one figure
import matplotlib.pyplot as plt
from PIL import Image

# width 9 subplots
fig, axs = plt.subplots(1, 9, figsize=(20, 15))  # figsize width=20

for i, threshold in enumerate(np.linspace(0, 0.6, 9)):
# for i, threshold in enumerate([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.15, 0.175, 0.2]):
    # add
    # pipe = DiTPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")
    # dfa_config = transform_model_dfa(pipe.transformer, n_steps=n_steps)

    # reset all compression methods to "raw"
    for step in range(n_steps):
        dfa_config.reset_step_method(step)
    latency = dfa_test_latency(pipe, calib_x, num_inference_steps=n_steps)
    # add
    unregister_refresh_stepi_hook(pipe.transformer)
    # compress_methods = fisher_info_planning(layer_compression_influences, dfa_config, threshold)
    update_layer_influence_new(pipe, dataloader, dfa_config, model_misc, alpha = threshold)
    print(dfa_config)
    unregister_refresh_stepi_hook(pipe.transformer)
    register_refresh_stepi_hook(pipe.transformer, n_steps)
    latency = dfa_test_latency(pipe, calib_x, num_inference_steps=n_steps)
    # generate a image
    generator = torch.Generator().manual_seed(seed)
    # height stack batch images
    images = pipe(calib_x, num_inference_steps=n_steps, generator=generator).images
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
plt.savefig(f"output/{model_id.replace('/', '_')}_fisher_info_planning.png")

# for layer_name in dfa_config.layer_names:
#     candidates = dfa_config.get_available_candidates(layer_name)
#     for step_i in range(1, n_steps):
#         # random select one candidate
#         choice = random.choice(candidates)
#         # choice = "output_share"
#         dfa_config.set_layer_step_method(layer_name, step_i, choice)
#         # print(f"Set {layer_name} step {step_i} to {choice}")
# dfa_config.apply_configs(verbose=True)
# latency = dfa_test_latency(pipe, calib_x, num_inference_steps=n_steps)
# print("dfa latency", latency)

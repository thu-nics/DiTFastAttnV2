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
)
import ditfastattn_api.models.dit_misc as dit_misc
import torch
import random
import numpy as np

model_id = "facebook/DiT-XL-2-512"
seed = 3
n_steps = 5
calib_x = torch.randint(0, 1000, (1,), generator=torch.Generator().manual_seed(seed)).to("cuda")
pipe = DiTPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")

n_samples = 1
model_misc = dit_misc


def dataloader():
    generator = torch.Generator().manual_seed(seed)
    for _ in range(n_samples):
        calib_x = torch.randint(0, 1000, (1,), generator=generator).to("cuda")
        yield [calib_x], {"num_inference_steps": n_steps}


pipe.transformer.to(torch.float32)
layer_gradients = get_layer_fisher_info(pipe, dataloader, model_misc)
# breakpoint()

pipe.transformer.to(torch.float16)
dfa_config = transform_model_dfa(pipe.transformer, n_steps=n_steps)
# latency = dfa_test_latency(pipe, calib_x, num_inference_steps=n_steps)
# print("raw latency", latency)

layer_compression_influences = get_compression_method_influence(
    pipe, dfa_config, dataloader, layer_gradients, model_misc
)

for threshold in np.linspace(0.1, 0.9, 9):
    unregister_refresh_stepi_hook(pipe.transformer)
    compress_methods = fisher_info_planning(layer_compression_influences, dfa_config, threshold)
    latency = dfa_test_latency(pipe, calib_x, num_inference_steps=n_steps)
    register_refresh_stepi_hook(pipe.transformer, n_steps)
    # generate a image
    generator = torch.Generator().manual_seed(seed)
    image = pipe(calib_x, num_inference_steps=n_steps, generator=generator).images[0]
    image.save(f"output/{model_id.replace('/', '_')}_threshold_{threshold}.png")
    print(f"threshold {threshold} latency {latency}")

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

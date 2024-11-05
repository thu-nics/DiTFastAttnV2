from ditfastattn_api.api import transform_model_dfa, dfa_test_latency
from diffusers import DiTPipeline, AutoPipelineForText2Image, StableDiffusion3Pipeline
import torch
import random
import json

model_id = "./models/stable-diffusion-3-medium-diffusers/"
seed = 3
n_steps = 5
resolution = 1024
calib_x = torch.randint(0, 1000, (1,), generator=torch.Generator().manual_seed(seed)).to("cuda")
pipe = StableDiffusion3Pipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")


with open(f"./data/mscoco/annotations/captions_val2014.json") as f:
    mscoco_anno = json.load(f)

caption_list = []
for i in calib_x:
    caption_list.append(mscoco_anno["annotations"][i]["caption"])

dfa_config = transform_model_dfa(pipe.transformer, n_steps=n_steps)
results = dfa_test_latency(pipe, caption_list, num_inference_steps=n_steps, height=resolution, width=resolution)
print("raw latency", results)

for layer_name in dfa_config.layer_names[:10]:
    candidates = dfa_config.get_available_candidates(layer_name)
    for step_i in range(1, n_steps):
        # random select one candidate
        choice = random.choice(candidates)
        choice = "raw"
        if "attn" in layer_name:
            choice = "residual_window_attn_128"
            choice = "output_share"
        dfa_config.set_layer_step_method(layer_name, step_i, choice)
        # print(f"Set {layer_name} step {step_i} to {choice}")
dfa_config.apply_configs(verbose=True)
results = dfa_test_latency(pipe, caption_list, num_inference_steps=n_steps, height=resolution, width=resolution)
print("dfa latency", results)

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

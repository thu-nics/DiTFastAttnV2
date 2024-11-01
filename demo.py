from ditfastattn_api.api import transform_model_dfa, dfa_test_latency
from diffusers import DiTPipeline, AutoPipelineForText2Image
import torch
import random

model_id = "facebook/DiT-XL-2-512"
seed = 3
n_steps = 5
calib_x = torch.randint(0, 1000, (1,), generator=torch.Generator().manual_seed(seed)).to("cuda")
pipe = DiTPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")

dfa_config = transform_model_dfa(pipe.transformer, n_steps=n_steps)
latency = dfa_test_latency(pipe, calib_x, num_inference_steps=n_steps)
print("raw latency", latency)

for layer_name in dfa_config.layer_names:
    candidates = dfa_config.get_available_candidates(layer_name)
    for step_i in range(1, n_steps):
        # random select one candidate
        choice = random.choice(candidates)
        # choice = "output_share"
        dfa_config.set_layer_step_method(layer_name, step_i, choice)
        # print(f"Set {layer_name} step {step_i} to {choice}")
dfa_config.apply_configs(verbose=True)
latency = dfa_test_latency(pipe, calib_x, num_inference_steps=n_steps)
print("dfa latency", latency)

from ditfastattn_api.api import transform_model_dfa, dfa_test_latency

# from diffusers import DiTPipeline, AutoPipelineForText2Image
from diffusers import Transformer2DModel, PixArtSigmaPipeline
import torch
import random
import json

model_id = "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS"
seed = 3
n_steps = 5

transformer = Transformer2DModel.from_pretrained(
    model_id,
    subfolder="transformer",
    torch_dtype=torch.float16,
    use_safetensors=True,
)

pipe = PixArtSigmaPipeline.from_pretrained(
    "PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers",
    transformer=transformer,
    torch_dtype=torch.float16,
    use_safetensors=True,
)
pipe.config._name_or_path = model_id

pipe.to("cuda")
calib_x = torch.randint(0, 1000, (1,), generator=torch.Generator().manual_seed(seed)).to("cuda")
with open(f"/nvme_data/yuanzhihang/docs/DiTFastAttention/data/mscoco/annotations/captions_val2014.json") as f:
    mscoco_anno = json.load(f)

caption_list = []
for i in calib_x:
    caption_list.append(mscoco_anno["annotations"][i]["caption"])
# pipe = DiTPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")
print(caption_list)

dfa_config = transform_model_dfa(pipe.transformer, n_steps=n_steps)
latency = dfa_test_latency(pipe, caption_list, num_inference_steps=n_steps)
print("raw latency", latency)

for layer_name in dfa_config.layer_names:
    candidates = dfa_config.get_available_candidates(layer_name)
    for step_i in range(1, n_steps - 1):
        # random select one candidate
        choice = random.choice(candidates)
        dfa_config.set_layer_step_method(layer_name, step_i, choice)
        # print(f"Set {layer_name} step {step_i} to {choice}")
dfa_config.apply_configs(verbose=True)
latency = dfa_test_latency(pipe, caption_list, num_inference_steps=n_steps)
print("dfa latency", latency)

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

dfa_config = transform_model_dfa(pipe.transformer, n_steps=n_steps)
print(dfa_config)
latency = dfa_test_latency(pipe, calib_x, num_inference_steps=n_steps)
print("raw latency", latency)

breakpoint()


def dataloader():
    generator = torch.Generator().manual_seed(seed)
    for _ in range(1):
        calib_x = torch.randint(0, 1000, (6,), generator=generator).to("cuda")
        yield [calib_x], {"num_inference_steps": n_steps}

for i, (args, kwargs) in enumerate(dataloader()):
    print(f">>> calibration fisher info sample {i} <<<")
    output_dict = dit_misc.inference_fn_with_output_record(pipe, *args, **kwargs)

for layer_name, value in dfa_config.layers:
    for t in range(len(dfa_config.layers['kwargs'])):
        dfa_config.set_layer_step_method(layer_name, t, "output_share")
        


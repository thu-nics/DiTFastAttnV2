DEFAULT_ATTN_CANDIDATES = ["raw", "output_share", "residual_window_attn_128", "residual_window_attn_256"]
# DEFAULT_ATTN_CANDIDATES = ["raw", "output_share"]
DEFAULT_FFN_CANDIDATES = ["raw", "output_share"]
import copy


class DiTFastAttnConfig:
    def __init__(self):
        self.layers = {}

    def add_attn_processor(self, layer_name, attn_processor, candidates=DEFAULT_ATTN_CANDIDATES):
        kwargs = {
            "steps_method": copy.deepcopy(attn_processor.steps_method),
        }
        self.layers[layer_name] = {"type": "attn", "obj": attn_processor, "candidates": candidates, "kwargs": kwargs}

    def add_ffn_layer(self, layer_name, ffn, candidates=DEFAULT_FFN_CANDIDATES):
        kwargs = {
            "steps_method": copy.deepcopy(ffn.steps_method),
        }
        self.layers[layer_name] = {"type": "ffn", "obj": ffn, "candidates": candidates, "kwargs": kwargs}

    def apply_configs(self, verbose=False):
        for layer_name, layer_info in self.layers.items():
            for key, value in layer_info["kwargs"].items():
                if verbose and value != layer_info["obj"].__getattribute__(key):
                    print(f"{layer_name}: Applying {key} = {value}")
            layer_info["obj"].update_config(**layer_info["kwargs"])

    def set_layer_step_method(self, layer_name, step_idx, method):
        self.layers[layer_name]["kwargs"]["steps_method"][step_idx] = method

    def set_layer_config(self, layer_name, **kwargs):
        for key, value in kwargs.items():
            if key not in self.layers[layer_name]["kwargs"]:
                raise ValueError(f"{key} is not a valid attribute for {layer_name}")
            self.layers[layer_name]["kwargs"][key] = value

    @property
    def layer_names(self):
        return list(self.layers.keys())

    def get_available_candidates(self, layer_name):
        return self.layers[layer_name]["candidates"]

    def __repr__(self):
        # print the layer names and the available candidates for each layer
        for layer_name, layer_info in self.layers.items():
            print(f"{layer_name}: \n   candidates: {layer_info['candidates']}\n   kwargs: {layer_info['kwargs']}")
        return ""

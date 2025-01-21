# DEFAULT_ATTN_CANDIDATES = ["output_share", "residual_window_attn_128", "residual_window_attn_256"]
DEFAULT_ATTN_CANDIDATES = ["output_share"]
# DEFAULT_ATTN_CANDIDATES = ["output_share"]
# DEFAULT_ATTN_CANDIDATES = ["raw", "output_share"]
DEFAULT_FFN_CANDIDATES = ["output_share"]
import copy


class DiTFastAttnConfig:
    def __init__(self):
        self.layers = {}
        self.latency = {}

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
        self.layers[layer_name]["obj"].update_config(steps_method=self.layers[layer_name]["kwargs"]["steps_method"])

    def set_layer_config(self, layer_name, **kwargs):
        for key, value in kwargs.items():
            if key not in self.layers[layer_name]["kwargs"]:
                raise ValueError(f"{key} is not a valid attribute for {layer_name}")
            self.layers[layer_name]["kwargs"][key] = value

    def get_layer_step_method(self, layer_name, step_idx):
        return self.layers[layer_name]['kwargs']['steps_method'][step_idx]

    def get_step_method(self, step_idx):
        method_dict = {}
        for layer_name in self.layers.keys():
            method_dict[layer_name] = self.get_layer_step_method(layer_name, step_idx)
        return method_dict

    # reset all layer method to raw for a designated step
    def reset_step_method(self, step_idx, method_dict=None):
        if method_dict == None:
            for layer_name in self.layers.keys():
                self.set_layer_step_method(layer_name, step_idx, "raw")
        else:
            for layer_name in self.layers.keys():
                self.set_layer_step_method(layer_name, step_idx, method_dict[layer_name])

    def display_step_method(self, step_idx):
        res = ""
        for layer_name in self.layers.keys():
            res = res + "name: " + layer_name + "method: " + self.get_layer_step_method(layer_name, step_idx) + "\n"
        print(res)

    def set_latency(self, latency_dict):
        self.latency = latency_dict

    def get_candidate_latency(self, candidate):
        return self.latency[candidate]

    @property
    def layer_names(self):
        return list(self.layers.keys())

    def get_available_candidates(self, layer_name):
        return self.layers[layer_name]["candidates"]

    def set_available_candidates(self, layer_name, candidates):
        self.layers[layer_name]['candidates'] = candidates
    
    def set_available_candidates_global(self, candidates):
        for layer_name in self.layers.keys():
            self.layers[layer_name]['candidates'] = candidates

    def __repr__(self):
        # print the layer names and the available candidates for each layer
        for layer_name, layer_info in self.layers.items():
            print(f"{layer_name}: \n   candidates: {layer_info['candidates']}\n   kwargs: {layer_info['kwargs']}")
        return ""
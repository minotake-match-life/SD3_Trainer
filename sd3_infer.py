# NOTE: Must have folder `models` with the following files:
# - `clip_g.safetensors` (openclip bigG, same as SDXL)
# - `clip_l.safetensors` (OpenAI CLIP-L, same as SDXL)
# - `t5xxl.safetensors` (google T5-v1.1-XXL)
# - `sd3_medium.safetensors` (or whichever main MMDiT model file)

import numpy as np
import sd3_impls
import torch
from other_impls import SD3Tokenizer, SDClipModel, SDXLClipG, T5XXLModel
from PIL import Image
from safetensors import safe_open
from sd3_impls import SDVAE

#################################################################################################
### Wrappers for model parts
#################################################################################################

def load_into(ckpt, model, prefix, device, dtype=None, remap=None, grad=False):
    """Just a debugging-friendly hack to apply the weights in a safetensors file to the pytorch module."""
    for key in ckpt.keys():
        model_key = key
        if remap is not None and key in remap:
            model_key = remap[key]
        if model_key.startswith(prefix) and not model_key.startswith("loss."):
            path = model_key[len(prefix) :].split(".")
            obj = model
            for p in path:
                if obj is list:
                    obj = obj[int(p)]
                else:
                    obj = getattr(obj, p, None)
                    if obj is None:
                        print(
                            f"Skipping key '{model_key}' in safetensors file as '{p}' does not exist in python model"
                        )
                        break
            if obj is None:
                continue
            try:
                tensor = ckpt.get_tensor(key).to(device=device)
                if dtype is not None and tensor.dtype != torch.int32:
                    tensor = tensor.to(dtype=dtype)
                if not grad:
                    # modify
                    obj.requires_grad_(False)
                # print(f"K: {model_key}, O: {obj.shape} T: {tensor.shape}")
                if obj.shape != tensor.shape:
                    print(
                        f"W: shape mismatch for key {model_key}, {obj.shape} != {tensor.shape}"
                    )
                # modify
                with torch.no_grad():
                    obj.set_(tensor)
            except Exception as e:
                print(f"Failed to load key '{key}' in safetensors file: {e}")
                raise e


CLIPG_CONFIG = {
    "hidden_act": "gelu",
    "hidden_size": 1280,
    "intermediate_size": 5120,
    "num_attention_heads": 20,
    "num_hidden_layers": 32,
}


class ClipG:
    def __init__(self, model_folder: str, device: str = "cpu"):
        with safe_open(
            f"{model_folder}/clip_g.safetensors", framework="pt", device="cpu"
        ) as f:
            self.model = SDXLClipG(
                CLIPG_CONFIG,
                device=device,
                dtype=torch.bfloat16
            )
            self.model.transformer.to(torch.bfloat16) # modify
            load_into(f, self.model.transformer, "", device, torch.bfloat16)


CLIPL_CONFIG = {
    "hidden_act": "quick_gelu",
    "hidden_size": 768,
    "intermediate_size": 3072,
    "num_attention_heads": 12,
    "num_hidden_layers": 12,
}


class ClipL:
    def __init__(self, model_folder: str):
        with safe_open(
            f"{model_folder}/clip_l.safetensors", framework="pt", device="cpu"
        ) as f:
            self.model = SDClipModel(
                layer="hidden",
                layer_idx=-2,
                device="cpu",
                dtype=torch.bfloat16, # was torch.float32
                layer_norm_hidden_state=False,
                return_projected_pooled=False,
                textmodel_json_config=CLIPL_CONFIG,
            )
            self.model.transformer.to(torch.bfloat16) # modify
            load_into(f, self.model.transformer, "", "cpu", torch.bfloat16)


T5_CONFIG = {
    "d_ff": 10240,
    "d_model": 4096,
    "num_heads": 64,
    "num_layers": 24,
    "vocab_size": 32128,
}

class T5XXL:
    def __init__(self, model_folder: str, device: str = "cpu", dtype=torch.float32):
        with safe_open(
            f"{model_folder}/t5xxl_fp16.safetensors", framework="pt", device="cpu"
        ) as f:
            self.model = T5XXLModel(T5_CONFIG, device=device, dtype=dtype)
            self.model.transformer.to(dtype=dtype) # modify
            load_into(f, self.model.transformer, "", device, dtype)

class VAE:
    def __init__(self, model, dtype: torch.dtype = torch.float16):
        with safe_open(model, framework="pt", device="cpu") as f:
            self.model = SDVAE(device="cpu", dtype=dtype).eval().cpu()
            prefix = ""
            if any(k.startswith("first_stage_model.") for k in f.keys()):
                prefix = "first_stage_model."
            load_into(f, self.model, prefix, "cpu", dtype)

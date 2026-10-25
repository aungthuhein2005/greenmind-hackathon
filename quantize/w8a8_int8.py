"""Data-free INT8 W8A8 quantization of the official MANGO1.5-Qwen3.5-9B BF16 weights.

INT8 per-channel weights + dynamic per-token INT8 activations (no calibration data needed), so the
A100's INT8 tensor cores do the matmuls. Quantized: MLP, full attention and the linear-attention
projections (in_proj_qkv, in_proj_z, out_proj). Kept in BF16: linear-attention gates (in_proj_a/b),
lm_head, the MTP head and the vision tower.

transformers drops the checkpoint's mtp.* tensors on load, so they are copied unchanged (BF16) from the
original shards into the output afterwards.

  ~/lc/bin/python quantize/w8a8_int8.py <bf16 snapshot dir> <output dir>
"""
import glob
import json
import os
import shutil
import sys

import torch
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoModelForImageTextToText

src, out = sys.argv[1], sys.argv[2]

model = AutoModelForImageTextToText.from_pretrained(src, dtype=torch.bfloat16, device_map="cpu")
recipe = QuantizationModifier(
    targets="Linear",
    scheme="W8A8",  # int8 weights per channel, int8 activations dynamic per token
    ignore=["lm_head", "re:.*in_proj_a$", "re:.*in_proj_b$", "re:.*mtp.*", "re:.*visual.*"],
)
oneshot(model=model, recipe=recipe)
model.save_pretrained(out, save_compressed=True)

# tokenizer, chat template and processor files, unchanged
for f in os.listdir(src):
    if not f.endswith(".safetensors") and f not in ("model.safetensors.index.json", "config.json") \
            and not os.path.exists(os.path.join(out, f)):
        shutil.copy(os.path.join(src, f), out)

# MTP head: copy the original BF16 tensors and register them in the index
mtp = {}
for shard in sorted(glob.glob(os.path.join(src, "*.safetensors"))):
    with safe_open(shard, "pt") as f:
        for k in f.keys():
            if k.startswith("mtp."):
                mtp[k] = f.get_tensor(k)
save_file(mtp, os.path.join(out, "model-mtp.safetensors"), metadata={"format": "pt"})

idx_path = os.path.join(out, "model.safetensors.index.json")
if os.path.exists(idx_path):
    idx = json.load(open(idx_path))
else:  # single-shard save: build an index for it
    with safe_open(os.path.join(out, "model.safetensors"), "pt") as f:
        idx = {"metadata": {}, "weight_map": {k: "model.safetensors" for k in f.keys()}}
idx["weight_map"].update({k: "model-mtp.safetensors" for k in mtp})
json.dump(idx, open(idx_path, "w"), indent=2)

cfg = json.load(open(os.path.join(out, "config.json")))
print("saved", out, "| mtp tensors copied:", len(mtp),
      "| quant method:", cfg.get("quantization_config", {}).get("quant_method"),
      "| mtp layers in config:", cfg.get("text_config", cfg).get("mtp_num_hidden_layers"))

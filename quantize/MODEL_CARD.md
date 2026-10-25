---
base_model: CMKL/MANGO1.5-Qwen3.5-9B
library_name: vllm
tags:
- int8
- w8a8
- compressed-tensors
- llm-compressor
---

# MANGO1.5-Qwen3.5-9B-W8A8-INT8

INT8 W8A8 quantization of [CMKL/MANGO1.5-Qwen3.5-9B](https://huggingface.co/CMKL/MANGO1.5-Qwen3.5-9B), made during the
GreenMind LLM Inference Optimization Hackathon (Oct 2026) from the official BF16 weights.

- **Weights:** INT8, symmetric, per output channel. **Activations:** INT8, symmetric, dynamic per token.
- **Data-free:** no calibration data was used (llm-compressor `QuantizationModifier`, scheme `W8A8`).
- **Quantized:** MLP (`gate/up/down_proj`), full attention (`q/k/v/o_proj`) and the Gated DeltaNet linear-attention
  projections (`in_proj_qkv`, `in_proj_z`, `out_proj`).
- **Kept in BF16:** linear-attention gates (`in_proj_a`, `in_proj_b`), `lm_head`, the MTP head (`mtp.*`, copied
  unchanged from the BF16 checkpoint) and the vision tower.
- Why INT8: the A100 has INT8 tensor cores but no FP8; vLLM serves this with `CutlassInt8ScaledMMLinearKernel`.

Build script: `quantize/w8a8_int8.py` in https://github.com/aungthuhein2005/greenmind-hackathon

Serving (vLLM 0.32, single A100 40GB):

```bash
vllm serve <this repo> --served-model-name CMKL/MANGO1.5-Qwen3.5-9B \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
  --max-model-len 49152 --max-num-seqs 64 --max-num-batched-tokens 2048 --long-prefill-token-threshold 1024 \
  --enable-prefix-caching --language-model-only \
  --speculative-config '{"method":"mtp","num_speculative_tokens":2}'
```

GreenMind Validation (100 requests): S 93.5, 96/100 requests within SLO, p95 TPOT 24 ms, accuracy factor 0.99.

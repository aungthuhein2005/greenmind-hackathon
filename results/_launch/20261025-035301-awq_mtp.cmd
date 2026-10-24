# profile=awq_mtp
400.00 W, 1410 MHz
vllm serve CMKL/MANGO1.5-Qwen3.5-9B-AWQ-W4A16-mtp --host 0.0.0.0 --port 8000 --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 --max-model-len 49152 --gpu-memory-utilization 0.92 --max-num-seqs 44 --max-num-batched-tokens 12288 --enable-chunked-prefill --language-model-only --enable-prefix-caching --kv-cache-dtype auto --optimization-level 2 --stream-interval 1 --scheduling-policy priority --middleware newest_first.NewestFirst --speculative-config \{\"method\":\"mtp\"\,\"num_speculative_tokens\":2\} 

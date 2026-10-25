#!/usr/bin/env bash
# Launch the vLLM server for one profile.
#   ./serve/serve.sh baseline        # default vllm serve, BF16 (the S=0 reference)
#   ./serve/serve.sh bf16_tuned      # BF16 + serving knobs
#   ./serve/serve.sh awq_mtp         # provided AWQ-W4A16 + MTP head (main bet)
#   ./serve/serve.sh awq             # AWQ without speculative decoding
#
# Override any knob via env, e.g.:
#   MAX_SEQS=96 SPEC_TOKENS=1 ./serve/serve.sh awq_mtp
# Every launch writes the exact command to results/_launch/<timestamp>.cmd so the
# repo shows what was actually run.
set -euo pipefail

PROFILE="${1:-awq_mtp}"

# The VM has no CUDA toolkit (nvcc), so FlashInfer's JIT-compiled sampler crashes engine
# warm-up. Use vLLM's PyTorch sampler; output is identical at temperature 0 (greedy).
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
PORT="${PORT:-8000}"
BF16_MODEL="CMKL/MANGO1.5-Qwen3.5-9B"
AWQ_MODEL="CMKL/MANGO1.5-Qwen3.5-9B-AWQ-W4A16-mtp"
# Our INT8 W8A8 model, built from BF16 by quantize/w8a8_int8.py (A100 INT8 tensor cores)
INT8_MODEL="${INT8_MODEL:-$HOME/models/MANGO1.5-Qwen3.5-9B-W8A8-INT8}"
INT8_NAMES=(--served-model-name MANGO1.5-Qwen3.5-9B-W8A8-INT8 "$BF16_MODEL")

# Serving knobs (sweep these)
MAX_LEN="${MAX_LEN:-49152}"          # 32K prompt + 8192 output + headroom
GPU_UTIL="${GPU_UTIL:-0.92}"
MAX_SEQS="${MAX_SEQS:-128}"          # concurrency cap: too high -> TPOT > 50ms at peak
BATCHED_TOKENS="${BATCHED_TOKENS:-8192}"  # chunked-prefill budget: trades TTFT vs TPOT
SPEC_TOKENS="${SPEC_TOKENS:-2}"      # MTP draft tokens: 1-3; more helps low load, hurts peak
SPEC_METHOD="${SPEC_METHOD:-mtp}"    # if vLLM rejects "mtp", try "qwen3_next_mtp" / check `vllm serve --help=speculative`
PREFIX_CACHE="${PREFIX_CACHE:-1}"    # hybrid Mamba/DeltaNet: if vLLM errors, set PREFIX_CACHE=0

# Required by the grader contract (tool calling + thinking)
CONTRACT=(--host 0.0.0.0 --port "$PORT"
          --enable-auto-tool-choice --tool-call-parser qwen3_coder
          --reasoning-parser qwen3)

TUNED=(--max-model-len "$MAX_LEN"
       --gpu-memory-utilization "$GPU_UTIL"
       --max-num-seqs "$MAX_SEQS"
       --max-num-batched-tokens "$BATCHED_TOKENS"
       --enable-chunked-prefill
       --language-model-only)            # grader is text-only: skip vision encoder, more KV memory
if [[ "$PREFIX_CACHE" == "1" ]]; then TUNED+=(--enable-prefix-caching); else TUNED+=(--no-enable-prefix-caching); fi
KV_DTYPE="${KV_DTYPE:-auto}"         # fp8 halves KV reads/memory for the attention layers
TUNED+=(--kv-cache-dtype "$KV_DTYPE")
OPT_LEVEL="${OPT_LEVEL:-2}"          # vLLM -O0..-O3: higher = longer startup, faster steady state
TUNED+=(--optimization-level "$OPT_LEVEL")
LONG_PREFILL="${LONG_PREFILL:-0}"    # >0: prompts longer than this are prefilled in pieces of this size
if [[ "$LONG_PREFILL" != "0" ]]; then TUNED+=(--long-prefill-token-threshold "$LONG_PREFILL"); fi
STREAM_INTERVAL="${STREAM_INTERVAL:-1}"  # tokens per streamed chunk; >1 cuts host overhead with many streams
TUNED+=(--stream-interval "$STREAM_INTERVAL")
# Hybrid prefix caching: match prefixes every N tokens instead of every 544-token block, and store a GDN
# state where requests diverge (both needed for sub-block hits on Mamba/GDN layers)
PREFIX_UNIT="${PREFIX_UNIT:-0}"
if [[ "$PREFIX_UNIT" != "0" ]]; then TUNED+=(--prefix-match-unit "$PREFIX_UNIT"); fi
if [[ "${MAMBA_CKPT:-0}" == "1" ]]; then TUNED+=(--enable-mamba-shared-prefix-checkpoint); fi
# ASYNC=0: synchronous scheduling, so the long-prefill threshold applies before the next chunk is queued
if [[ "${ASYNC:-auto}" == "0" ]]; then TUNED+=(--no-async-scheduling); fi
# REASONING_END: text forced when the thinking budget (THINK_BUDGET, see newest_first.py) runs out.
# A transition phrase before </think> helps the model answer directly instead of reasoning in the answer.
case "${REASONING_END_PRESET:-}" in   # space-free names for sweep files
  concise) REASONING_END="I have enough information. Now I give only the concise final answer.</think>" ;;
  vllm)    REASONING_END="I have to give the solution based on the reasoning directly now.</think>" ;;
esac
if [[ -n "${REASONING_END:-}" ]]; then
  TUNED+=(--reasoning-config "{\"reasoning_end_str\": \"${REASONING_END}\"}")
fi

# Scheduling: fcfs (vLLM default), newest (newest-first, serve/newest_first.py)
# or slo (deadline-aware: feasible earliest-deadline-first, missed oldest-first, serve/slo_scheduler.py)
SCHED="${SCHED:-fcfs}"
if [[ "$SCHED" != "fcfs" ]]; then
  export PYTHONPATH="$(cd "$(dirname "$0")" && pwd)${PYTHONPATH:+:$PYTHONPATH}"
fi
case "$SCHED" in
  newest) TUNED+=(--scheduling-policy priority --middleware newest_first.NewestFirst) ;;
  shortest) export PRIORITY_MODE=shortest
          TUNED+=(--scheduling-policy priority --middleware newest_first.NewestFirst) ;;
  slo)    TUNED+=(--scheduling-policy priority --scheduler-cls slo_scheduler.SLOScheduler) ;;
esac

SPEC=(--speculative-config "{\"method\":\"$SPEC_METHOD\",\"num_speculative_tokens\":$SPEC_TOKENS}")

case "$PROFILE" in
  baseline)   CMD=(vllm serve "$BF16_MODEL" "${CONTRACT[@]}") ;;
  bf16_tuned) CMD=(vllm serve "$BF16_MODEL" "${CONTRACT[@]}" "${TUNED[@]}") ;;
  awq)        CMD=(vllm serve "$AWQ_MODEL"  "${CONTRACT[@]}" "${TUNED[@]}") ;;
  awq_mtp)    CMD=(vllm serve "$AWQ_MODEL"  "${CONTRACT[@]}" "${TUNED[@]}" "${SPEC[@]}") ;;
  int8_mtp)   CMD=(vllm serve "$INT8_MODEL" "${INT8_NAMES[@]}" "${CONTRACT[@]}" "${TUNED[@]}" "${SPEC[@]}") ;;
  int8)       CMD=(vllm serve "$INT8_MODEL" "${INT8_NAMES[@]}" "${CONTRACT[@]}" "${TUNED[@]}") ;;
  *) echo "unknown profile: $PROFILE" >&2; exit 1 ;;
esac

mkdir -p results/_launch
STAMP="$(date +%Y%m%d-%H%M%S)"
{ echo "# profile=$PROFILE"; nvidia-smi --query-gpu=power.limit,clocks.max.sm --format=csv,noheader; printf '%q ' "${CMD[@]}"; echo; } \
  | tee "results/_launch/$STAMP-$PROFILE.cmd"

exec "${CMD[@]}"

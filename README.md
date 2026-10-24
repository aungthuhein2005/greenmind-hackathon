# GreenMind — on-time LLM work per joule

Serving `CMKL/MANGO1.5-Qwen3.5-9B` on one A100 40GB so that as many answers as possible meet the
SLO per GPU joule. GreenMind LLM Inference Optimization Hackathon, Oct 2026.

**Weights:** we used only the organizer-provided originals —
[`CMKL/MANGO1.5-Qwen3.5-9B`](https://huggingface.co/CMKL/MANGO1.5-Qwen3.5-9B) (baseline) and
[`CMKL/MANGO1.5-Qwen3.5-9B-AWQ-W4A16-mtp`](https://huggingface.co/CMKL/MANGO1.5-Qwen3.5-9B-AWQ-W4A16-mtp) (submitted).
No derived weights were created.

## Result

**Best Final score: S = 48.6** (205/300 requests on time, 2.52x the baseline's on-time answers per joule, 167 J/request).

Every grader **Final** submission (300 requests), in order:

| # | Config | S | A | sE | sT | sL | Efficiency vs baseline | J / request | p95 TTFT | p95 TPOT | SLO met |
|---|---|---|---|---|---|---|---|---|---|---|---|
| – | Baseline (default `vllm serve`, BF16) | 0 | 1.00 | 0 | – | – | 1.00x | – | – | – | – |
| 1 | AWQ + MTP, prefix cache, 64 seqs, FCFS | 0.0 | 0.98 | 0.00 | 0.02 | 0.27 | 0.86x | 207 | 6.2 s | 62 ms | 87/300 |
| 2 | + newest-first scheduling, 48 seqs | 33.2 | 1.00 | 0.44 | 0.37 | 0.12 | 1.63x | 194 | 41.9 s | 51 ms | 154/300 |
| 3 | + newest-first scheduling, 40 seqs | 40.0 | 1.00 | 0.60 | 0.41 | 0.12 | 1.94x | 179 | 50.6 s | 44 ms | 169/300 |
| 4 | #3 + prefill chunk 16384 | 29.0 | 0.95 | 0.40 | 0.38 | 0.11 | 1.56x | 214 | 50.0 s | 44 ms | 163/300 |
| 5 | #3 with MTP off | 36.3 | 1.00 | 0.50 | 0.45 | 0.11 | 1.73x | 208 | 49.2 s | 42 ms | 176/300 |
| 6 | #3 with 350 W power cap | 32.8 | 1.00 | 0.48 | 0.33 | 0.10 | 1.69x | 179 | 57.2 s | 47 ms | 148/300 |
| 7 | #3 with 44 seqs | 40.9 | 1.00 | 0.60 | 0.39 | 0.13 | **1.94x** | **172** | 41.9 s | 44 ms | 163/300 |
| 8 | #7 + long-prefill threshold 2048, async scheduling off | 41.7 | 1.00 | 0.58 | 0.50 | 0.13 | 1.89x | 207 | 46.9 s | 48 ms | 191/300 |
| 9 | #8 re-run (same config) | 35.2 | 1.00 | 0.47 | 0.46 | 0.11 | 1.68x | 212 | 51.6 s | 49 ms | 174/300 |
| 10 | **#8 + thinking budget 1024 tokens (best)** | **48.6** | 0.90 | **0.84** | **0.54** | 0.14 | **2.52x** | **167** | 32.0 s | 48 ms | **205/300** |

Runs 4-6 each looked better than #3 in a single local run and lost on the grader; why is in [FAILED.md](FAILED.md).
Grader run-to-run noise (about ±10% energy, ±20 on-time requests) is larger than those effects. From then on we
only spent a Final on a candidate that beat the baseline across **repeated cold-cache runs**: 44 seqs (median
170 vs 157 on time over 3 vs 9 local runs) was the first to pass, and became run 7. A long-prefill threshold of
2048 tokens with async scheduling off (median 186 vs 170 on time) was the second, and became run 8: 28 more
on-time answers and the best peak share (sT 0.50), at about 20% more energy per request from synchronous
scheduling. The threshold only takes effect without async scheduling's look-ahead (vLLM issue #60476).

**Thinking budget (run 10).** The real bottleneck is output length: ~950 tokens per answer, mostly reasoning. With the
organizers' approval (GPU mentor on Discord: "Yes, you can cap the thinking budget") the server caps reasoning at
1024 tokens through vLLM's `thinking_token_budget`; thinking stays on for every request and the model then writes its
answer. Output fell 22%, on-time answers rose to 205/300 and energy to 167 J/request (2.52x baseline efficiency). The
cost is accuracy: R 0.955 on that run (A 0.90), from shorter reasoning on math/long items plus normal run-to-run noise.
Budget 512 lost too much math accuracy locally; 256 made the model reason inside the answer instead.

On the lighter 100-request Validation trace (no overload) the AWQ + MTP setup with first-come-first-served scored
48.7 (1.74x). The accuracy factor stays at about 1.00 on every run (R 0.96-1.05): 4-bit weights cost nothing
measurable on the grader's six classes.

## Final serving command

```bash
# GPU at its default 400 W limit (see FAILED.md for why we did not cap it)
VLLM_USE_FLASHINFER_SAMPLER=0 PYTHONPATH=serve \
vllm serve CMKL/MANGO1.5-Qwen3.5-9B-AWQ-W4A16-mtp --host 0.0.0.0 --port 8000 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
  --max-model-len 49152 --gpu-memory-utilization 0.92 \
  --max-num-seqs 44 --max-num-batched-tokens 8192 --enable-chunked-prefill \
  --long-prefill-token-threshold 2048 --no-async-scheduling \
  --language-model-only --enable-prefix-caching --kv-cache-dtype auto \
  --speculative-config '{"method":"mtp","num_speculative_tokens":2}' \
  --scheduling-policy priority --middleware newest_first.NewestFirst
# + THINK_BUDGET=1024 in the environment: the middleware sets thinking_token_budget on each request
```

Same thing via our script: `PREFIX_CACHE=1 MAX_SEQS=44 SCHED=newest LONG_PREFILL=2048 ASYNC=0 THINK_BUDGET=1024 ./serve/serve.sh awq_mtp`.
vLLM 0.32.0, installed by `serve/install.sh`. The middleware is [serve/newest_first.py](serve/newest_first.py)
(~40 lines, in-process); organizers confirmed server-side prioritization is allowed.

## Why it works

**1. Free memory so more requests fit.** The baseline is not short of power, it is short of memory. With BF16
weights (17.6 GB) only ~14 GB is left for KV cache; at the 4 req/s peak the cache hits 99%, ~75 requests run while
100+ queue, and almost nothing meets the SLO. Every late answer still burns joules.

| Change | Effect |
|---|---|
| **AWQ W4A16 weights** | 17.6 -> 7.7 GB of weights, KV cache 13.9 -> 23 GB (+66%); same accuracy |
| **MTP speculative decoding, 2 drafts** | ~2.2-2.9 tokens accepted per step -> fewer forward passes per answer |
| **`--language-model-only`** | grader is text-only; skips the vision encoder and its memory |
| **Prefix caching on** | grader traffic reuses tool definitions and multi-turn history; turning it off cost 213 -> 279 J/request |

**2. Serve newest-first under overload.** Even so, the Final trace's peak sends work faster than one A100 can
finish it. First-come-first-served then makes the queue grow until *every* new request misses its 0.5 s TTFT, so
the whole peak is wasted energy (Final: 87/300 on time, sT 0.02). Under overload the only requests that can still
be on time are the ones that just arrived, so we admit those first. Requests already stuck in the queue were late
anyway; they still complete (no drops, accuracy unchanged) and drain in the quiet phase. On-time answers:
87 -> 169 of 300 for the same hardware.

**3. Cap concurrency to the SLO.** With newest-first, fewer simultaneous decodes keep every admitted request
under the 50 ms TPOT SLO (64 -> 48 -> 40 seqs: p95 TPOT 59 -> 45 -> 41 ms locally, 51 -> 44 ms on the grader).

The cost is a long tail for the requests that were already late (p95 TTFT ~50 s, sL 0.12). We tried an SLO-aware
scheduler that serves missed requests oldest-first to shorten that tail; it lost more on-time answers than it
gained in latency (FAILED.md).

Every experiment, including the ones that made things worse, is in [EXPERIMENTS.md](EXPERIMENTS.md); the failures
and our explanations are in [FAILED.md](FAILED.md).

## Repository

```
serve/       install.sh, serve.sh (profiles), newest_first.py (scheduling middleware), slo_scheduler.py
             (tried, lost), power.sh, sweep_configs.sh, sweep_power.sh, sweeps/
harness/     workload.py (open-data workload), run.py (load/accuracy runner + NVML energy), score.py (S estimate)
results/     one folder per run: requests.jsonl, items.jsonl, power.csv, summary.json, score.txt; _launch/ = exact commands
tests/       mock_server.py to test the harness without a GPU
```

### Local harness

`harness/workload.py` builds a gradable workload with the grader's class mix from open data only
(MMLU, GSM8K, Belebele `tha_Thai`, synthetic needle-in-documents and fake-tool agentic episodes).
`harness/run.py load` replays it open-loop (ramp -> 4 req/s peak -> 1 req/s quiet) with thinking on,
`max_tokens` 8192, temperature 0, and reads GPU joules from NVML. `harness/score.py` applies the briefing's formulas.

What it got right and wrong, compared with the grader:
- Energy efficiency and accuracy tracked the grader closely (1.86x local vs 1.76x grader).
- Its 300-item run reproduces the **Final** trace's overload (we first mistook this for being "too harsh",
  because the 100-request Validation trace never overloads the server). It found newest-first scheduling
  before we spent a Final submission on it.
- It had no shared prefixes, so it could not see the value of prefix caching — the grader could.
- Replaying the same prompts with prefix caching on made later runs look far too good (warm cache);
  local runs must restart the server between configs.

```bash
./serve/install.sh                                         # uv venv + vLLM + deps
python harness/workload.py --out data/workload.jsonl       # 300 items
./serve/serve.sh awq_mtp                                   # terminal 1
python harness/run.py load --name my-run                   # terminal 2
python harness/score.py results/my-run --baseline results/baseline
```

Harness self-test without a GPU: `python tests/mock_server.py`, then
`python harness/workload.py --out data/mock.jsonl --synthetic --scale 0.2` and
`python harness/run.py load --name mock --workload data/mock.jsonl --base-url http://127.0.0.1:8765/v1 --peak-rate 20`.

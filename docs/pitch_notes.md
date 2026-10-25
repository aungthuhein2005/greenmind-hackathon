# Pitch notes (7 min talk + 7 min Q&A)

Round 2 is 60% of the final score: technical approach & trade-offs 40%, sustainability impact 40%, clarity &
delivery 20%. Numbers are from the grader's **Final** track (300 requests) unless marked otherwise.

## One-sentence story
"An LLM server that falls behind wastes energy on answers that arrive too late. We fixed the queue with
newest-first scheduling, cut runaway reasoning with an approved thinking cap, and built our own INT8 model for
the A100's INT8 tensor cores — taking the score from 0 to 99.7 on the same GPU."

## The progression (the spine of the talk)

| Step | Final S | On time | Efficiency | What changed |
|---|---|---|---|---|
| Memory fixes only, first-come-first-served | 0.0 | 87/300 | 0.86x | AWQ 4-bit + MTP; the peak queue made every new request late |
| + newest-first, 44 seqs, long-prefill threshold | 41.7 | 191/300 | 1.89x | serve the request that can still be on time |
| + thinking budget 1024 (organizer-approved) | 48.6 | 205/300 | 2.52x | fewer tokens per answer; accuracy dipped (A 0.90) |
| **Our INT8 W8A8 model** + MTP2, 64 seqs | 95.9 | 279/300 | 2.84x | INT8 tensor cores: the backlog disappears |
| + thinking budget 2048 | 98.6 | 286/300 | 3.80x | energy at full marks, accuracy *up* (R 1.063) |
| **+ 72 concurrent sequences** | **99.7** | **295/300** | **3.96x** | peak share 0.93 -> 0.98 |

Best run: A 1.00, sE 1.00, sT 0.98, sL 1.00; p95 first token 1.5 s, p95 time per token 34 ms, 152 J/request.

## Slide flow (~10 slides, ~40 s each)

1. **The problem.** Late, failed or cut-off answers still burn energy but count for nothing. Measure joules per
   *useful* answer.
2. **Diagnosis.** Default BF16 server: 17.6 GB of weights, KV cache 99% full at the 4 req/s peak, 100+ queued.
3. **Fix 1 — memory.** AWQ 4-bit (organizers' model), MTP speculative decoding, text-only mode, prefix caching.
   48.7 on the light Validation trace — then **0.0 on Final**: the heavy peak still overloaded the GPU.
4. **Fix 2 — newest-first.** Under overload, only a request that just arrived can still meet its deadline; serve
   it first. ~40 lines of in-process vLLM middleware; organizers confirmed prioritization is allowed. 0 -> 41.7.
5. **Fix 3 — our own INT8 model.** AWQ stores 4-bit weights but multiplies in 16-bit; the A100 has INT8 tensor
   cores (~2x the 16-bit rate) and no FP8. We built a data-free INT8 W8A8 model from the official BF16 weights
   with llm-compressor (MLP, attention and Gated-DeltaNet projections in INT8; gates, lm_head, MTP head in BF16).
   41.7 -> 95.9; p95 first token 32 s -> 1.3 s.
6. **Fix 4 — thinking cap.** Organizer-approved `thinking_token_budget` of 2048: thinking stays on, runaway chains
   stop. Energy -23% (201 -> 154 J/request), accuracy *rose* (R 1.063) because fewer answers hit the 8,192-token
   limit. 95.9 -> 98.6. Then 72 concurrent sequences: 99.7.
7. **Results.** The progression table above, as a bar chart.
8. **What didn't work** (judges score this): power caps (250 W -> 0.0), prefix caching off (-19 points), big
   concurrency on AWQ (34-42 on time), deadline-aware scheduler, shortest-prompt-first (fewer on time), thinking
   cap 512/256 (math accuracy, spill into the answer), 80 seqs (no gain over 72). All in FAILED.md.
9. **Sustainability at scale** (state assumptions on the slide):
   - 3.96x on-time answers per joule = **~75% less energy per useful answer** vs the baseline.
   - Per on-time answer: ours ≈ 155 J (152 J/request ÷ 98.3% on time) vs ≈ 614 J for the baseline.
   - For 1 million on-time answers a day: ~128 kWh/day saved, **~47 MWh/year**; at ~0.4 kg CO2/kWh (Thai grid,
     approximate) ≈ **19 t CO2/year** per million daily answers.
   - Same GPU, a model derived from the official weights, software only; accuracy at the baseline level (Thai included).
10. **Takeaways.** (1) Measure joules per on-time answer. (2) Match the number format to the GPU — INT8 on Ampere.
    (3) Under overload, serve newest-first and cap concurrency to the SLO. (4) Bound reasoning length. (5) Test on
    realistic traffic with a cold cache, and repeat runs before trusting a small gain.

## Speak the organizers' language
Their lecture ends with **Measure -> Hypothesize -> Change one thing -> Re-measure -> Explain** and "keep a failure
log". We did exactly that: EXPERIMENTS.md (every run) and FAILED.md (every failure with its explanation).

| Lecture module | Our evidence |
|---|---|
| 02 KV cache limits batch size | baseline KV 99% full at peak |
| 03 Quantization: W4A16 vs W8A8, INT8 on Ampere | AWQ capped us at ~49; our INT8 W8A8 reached 99.7 |
| 03 Speculative decoding | MTP2 kept throughout (acceptance ~2.9 tokens/step with INT8) |
| 04 Prefix caching on hybrid layers | turning it off cost 213 -> 279 J/request on real agent traffic |
| 05 Scheduler order / unbounded queues | newest-first: Final 0 -> 41.7 |
| 06 Largest batch that meets the latency target | 64 -> 72 seqs: peak share 0.93 -> 0.98 at 34 ms TPOT |
| 08 Power limits | the exception: compute-bound MTP decode; 250 W scored 0 |
| 09 Benchmark realistically | local harness lacked shared prefixes / cold cache; both bit us and are documented |

## Likely Q&A
- *Is the thinking cap allowed?* Yes — the organizers' GPU mentor confirmed on Discord ("Yes, you can cap the
  thinking budget"). Thinking stays on for every request; max_tokens, temperature and sampling are untouched.
- *Did INT8 cost accuracy?* No: A = 1.00 on every INT8 Final; R 0.991 without the cap, 1.025-1.063 with it.
- *Why not FP8?* The A100 (Ampere) has no FP8 tensor cores; INT8 W8A8 is the fastest supported path.
- *Why no calibration data?* Weights are per-channel INT8 and activations are quantized dynamically per token, so
  no activation statistics are needed; it took ~1 minute on CPU and kept accuracy.
- *Where are the weights?* https://huggingface.co/aungthuhein-dev/MANGO1.5-Qwen3.5-9B-W8A8-INT8, built during the
  event from CMKL/MANGO1.5-Qwen3.5-9B; script `quantize/w8a8_int8.py`.
- *Isn't newest-first unfair?* Under overload, old requests would miss their deadline under FIFO too; they still
  complete with the same accuracy. With INT8 the backlog is gone and p95 first token is 1.5 s.
- *Why did the power cap fail?* MTP verifies several tokens per step, so decode is compute-bound; lower clocks
  slowed every token past the 50 ms limit.
- *What's left?* Only the peak share (sT 0.98, ~5 late requests). Grader run-to-run noise (99.6 vs 99.7 at
  near-identical settings) is about the size of the remaining gap.

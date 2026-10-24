# Pitch notes (7 min talk + 7 min Q&A)

Due with the repo link and HF statement by **Oct 25, 11:30** (via the organizers' submit form).
Round 2 is 60% of the final score:
technical approach & trade-offs 40%, sustainability impact 40%, clarity & delivery 20%.

Numbers are from the grader's **Final** track (300 requests) unless marked otherwise. Update the headline if a
later Final beats 40.0.

## One-sentence story
"When an LLM server is overloaded, first-come-first-served makes *every* answer late, so all that energy is wasted;
we free memory with 4-bit weights and then serve the newest request first, which nearly doubles the useful answers
per joule on the same GPU."

## Speak the organizers' language
Their lecture ("Inference Optimization 101") ends with **Measure -> Hypothesize -> Change one thing -> Re-measure
-> Explain** and "keep a failure log". Say that we followed it, and show EXPERIMENTS.md / FAILED.md as proof.
Map each result to their module:

| Lecture module | Our evidence |
|---|---|
| 02 KV cache "often decides how many requests fit" | baseline KV cache 99% full at peak; AWQ freed +66% |
| 03 Quantization: W4A16 needs no native INT4; FP8 needs Hopper | why AWQ on A100, not FP8 |
| 03 Speculative decoding "hurts at high batch" | MTP tested on and off under the Final overload (see EXPERIMENTS.md) |
| 04 Prefix caching "works differently" on hybrid layers | looked useless locally, cost 213 -> 279 J/req when off on real traffic |
| 05 Scheduler: order (FCFS vs priority), "unbounded queues hurt under load" | **newest-first: Final 0 -> 40.0** — our main contribution |
| 06 "Largest batch that still meets your latency target" | 40 concurrent requests -> p95 TPOT 44 ms (< 50) |
| 08 Power limits: "a modest slowdown can save a lot" | the exception: MTP decode is compute-bound, 250 W scored 0 |
| 09 "Benchmark realistically" | our harness lacked shared prefixes and warm/cold cache control — both bit us, both documented |

## Slide flow (~8 slides, ~50 s each)

**1. The problem.** Serving is tuned for speed; nobody measures joules per *useful* answer. An answer that arrives
late still burns energy but counts for nothing.

**2. What we found: one A100 can't keep up with the peak.**
- Default BF16 server: weights take 17.6 GB of 40 GB, KV cache fills to 99% at the 4 req/s peak, 100+ requests queue.
- Even after our memory fixes, first-come-first-served scored **0** on Final: 87/300 on time, peak phase 2% on time.
  The queue grows until every new request misses its 0.5 s first-token deadline.

**3. Fix 1 — free memory and do less work per token** (one line each)
- AWQ 4-bit weights: 17.6 -> 7.7 GB, KV cache +66%
- MTP speculative decoding (2 drafts): ~2.2-2.9 tokens per forward pass
- Text-only mode, prefix caching (tool schemas and multi-turn history are reused)

**4. Fix 2 — serve the newest request first** (the original idea; show a queue diagram)
- Under overload, only a request that just arrived can still meet its deadline. Serve those first.
- Requests already stuck were late anyway; they still finish (no drops, same accuracy) in the quiet phase.
- ~40 lines of in-process vLLM middleware using its priority scheduler; organizers confirmed it's allowed.
- Then cap concurrency to the SLO: 40 simultaneous requests keeps p95 time-per-token at 44 ms (< 50 ms).

**5. Results (grader Final)**

| | FCFS (our memory fixes only) | + newest-first, 48 | **+ newest-first, 40** |
|---|---|---|---|
| Score S | 0.0 | 33.2 | **40.0** |
| On-time answers | 87/300 | 154/300 | **169/300** |
| Efficiency vs baseline | 0.86x | 1.63x | **1.94x** |
| Energy per request | 207 J | 194 J | **179 J** |
| p95 time per token | 62 ms | 51 ms | **44 ms** |
| Accuracy factor A | 0.98 | 1.00 | 1.00 |

On the lighter Validation trace (100 requests, no overload) the memory fixes alone scored 48.7.

**6. What didn't work — and what it taught us** (judges explicitly score this)
- **Power cap 250 W: 48.7 -> 0 (Validation).** MTP makes decoding compute-bound, so lower clocks slowed every token;
  on-time answers fell 57 -> 33. Lesson: here, speed *is* efficiency.
- **Prefix caching off: 48.7 -> 29.6.** Our test traffic had no shared prefixes so it looked useless; real agent
  traffic depends on it. Lesson: benchmark on realistic traffic.
- **Smaller prefill chunks: 48.7 -> 32.6.** Fixed the latency tail but 4x more forward passes cost more energy.
- **Smarter deadline-aware scheduler:** shortened the latency tail a little but lost ~15% of on-time answers.
  Newest-first is the better trade for energy.
- 3 MTP drafts, 128/64/32 concurrency: all worse (FAILED.md).

**7. Sustainability at scale** (state assumptions on the slide)
- 1.94x more on-time answers per joule = **~48% less energy per useful answer**, on existing hardware, no accuracy loss.
- Per on-time answer: ours ≈ 318 J (179 J/request ÷ 56% on time) vs ≈ 617 J for the baseline.
- For 1 million on-time answers a day: ~83 kWh/day saved, **~30 MWh/year**; at ~0.4 kg CO2/kWh (Thailand grid,
  approximate) ≈ **12 t CO2/year** per million daily answers.
- Newest-first is a software change any vLLM deployment can adopt; it matters most exactly when servers are
  overloaded, which is when the most energy is being wasted.
- Thai stays as accurate as the baseline: efficiency without sacrificing the local language.

**8. Takeaways anyone can copy tomorrow**
1. Measure joules per *on-time* answer, not tokens/s.
2. Quantize weights to free KV memory before buying GPUs.
3. Under overload, FIFO wastes the peak: admit newest-first and cap concurrency to the SLO.
4. Don't power-cap a compute-bound decoder; test with realistic traffic and a cold cache.

## Likely Q&A
- *Isn't newest-first unfair to old requests?* They still complete with the same accuracy; under overload they'd
  miss the deadline under FIFO too. In production you'd add an age cap; we tried a deadline-aware variant (FAILED.md).
- *Why did your latency score (sL 0.12) drop?* The p95 first-token time is set by the backlog that was late anyway.
  Energy is 60% of the score, latency 20%; we chose the trade on purpose and measured the alternative.
- *Why AWQ, not FP8?* A100 (Ampere) has no FP8 tensor cores; FP8 would only shrink memory. AWQ 4-bit kept accuracy.
- *Does power capping ever work?* When decode is memory-bound (no speculative decoding). With MTP it's compute-bound.
- *Would this hold at 10x traffic?* Newest-first keeps the fresh share on time at any overload; capacity beyond that
  needs more replicas, each with the same efficiency.
- *How sure are you about accuracy?* Grader R 0.98-1.05 across all runs, A = 1.00 on every best run.

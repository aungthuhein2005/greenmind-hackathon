"""Pure-decode capacity: N concurrent short prompts, each forced to generate exactly --tokens tokens.
Reports aggregate tokens/s and per-request TPOT, so we can see how decode speed scales with batch
size without prefill interference.

  python harness/decode_bench.py --concurrency 8 32 64 128 192 --tokens 512 --tag awq_mtp
"""
import argparse
import asyncio
import json
import os
import statistics
import sys
import time

import httpx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run import Energy  # noqa: E402


async def one(client, base, model, i, tokens):
    body = {"model": model, "stream": True, "stream_options": {"include_usage": True},
            "messages": [{"role": "user", "content": f"Write a long story about city number {i}."}],
            "max_tokens": tokens, "min_tokens": tokens, "ignore_eos": True, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False}}
    t0, first, last, n = time.perf_counter(), None, None, 0
    async with client.stream("POST", f"{base}/chat/completions", json=body, timeout=900) as r:
        async for line in r.aiter_lines():
            if not line.startswith("data:") or line.endswith("[DONE]"):
                continue
            ch = json.loads(line[5:])
            if ch.get("usage"):
                n = ch["usage"]["completion_tokens"]
            for c in ch.get("choices") or []:
                d = c.get("delta") or {}
                if d.get("content") or d.get("reasoning_content") or d.get("reasoning"):
                    now = time.perf_counter()
                    first = first or now
                    last = now
    return {"ttft": first - t0, "tpot": (last - first) / max(1, n - 1), "n": n}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concurrency", type=int, nargs="+", default=[8, 32, 64, 128, 192])
    ap.add_argument("--tokens", type=int, default=512)
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    async with httpx.AsyncClient(limits=httpx.Limits(max_connections=1000), timeout=900) as client:
        model = (await client.get(f"{a.base_url}/models")).json()["data"][0]["id"]
        await one(client, a.base_url, model, 0, 32)  # warm-up
        for c in a.concurrency:
            os.makedirs("results/_bench", exist_ok=True)
            e = Energy(f"results/_bench/{a.tag}-c{c}.power.csv")
            e.start()
            t0 = time.perf_counter()
            res = await asyncio.gather(*(one(client, a.base_url, model, i, a.tokens) for i in range(c)))
            wall = time.perf_counter() - t0
            j = e.stop()
            toks = sum(r["n"] for r in res)
            tp = sorted(r["tpot"] for r in res)
            line = (f"{a.tag} c={c:<4} tok/s {toks / wall:7.0f}  TPOT median {1000 * statistics.median(tp):5.1f} ms"
                    f"  p95 {1000 * tp[int(0.95 * (len(tp) - 1))]:5.1f} ms  mJ/token {1000 * j / toks:6.1f}  avg W {j / wall:4.0f}")
            print(line, flush=True)


if __name__ == "__main__":
    asyncio.run(main())

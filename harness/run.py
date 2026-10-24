"""Replay the workload against an OpenAI-compatible endpoint, recording per-request
TTFT / TPOT / tokens / finish reason and total GPU joules from NVML.

  python harness/run.py load --name awq_mtp-s128          # open-loop ramp -> peak -> quiet (grader-like)
  python harness/run.py acc  --name awq_mtp-acc -c 48     # closed loop, fastest way to measure accuracy

Run it ON THE VM (NVML reads the local GPU). It uses CPU only, so it does not disturb the GPU.
Outputs results/<name>/{requests.jsonl, items.jsonl, power.csv, summary.json}; then run score.py.
"""
import argparse
import asyncio
import json
import os
import sys
import threading
import time

import httpx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from workload import call_tool  # noqa: E402

MAX_TOKENS = 8192
TIMEOUT_S = 900


# ---------------------------------------------------------------- energy
class Energy:
    """Total-energy counter (mJ) + 0.5 s power samples. Falls back to 0 J off-GPU."""

    def __init__(self, csv_path, period=0.5):
        self.csv_path, self.period, self.nv = csv_path, period, None
        try:
            import pynvml
            pynvml.nvmlInit()
            self.nv, self.h = pynvml, pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception as e:  # noqa: BLE001
            print(f"[energy] NVML unavailable ({e}); joules will read 0")

    def joules(self):
        return self.nv.nvmlDeviceGetTotalEnergyConsumption(self.h) / 1000.0 if self.nv else 0.0

    def power_limit_w(self):
        return self.nv.nvmlDeviceGetEnforcedPowerLimit(self.h) / 1000.0 if self.nv else None

    def _sample(self):
        with open(self.csv_path, "w") as f:
            f.write("t,watts,sm_mhz,util\n")
            while not self._stop.is_set():
                if self.nv:
                    w = self.nv.nvmlDeviceGetPowerUsage(self.h) / 1000.0
                    mhz = self.nv.nvmlDeviceGetClockInfo(self.h, self.nv.NVML_CLOCK_SM)
                    util = self.nv.nvmlDeviceGetUtilizationRates(self.h).gpu
                    f.write(f"{time.perf_counter() - self.t0:.2f},{w:.1f},{mhz},{util}\n")
                self._stop.wait(self.period)

    def start(self):
        self.t0, self.j0 = time.perf_counter(), self.joules()
        self._stop = threading.Event()
        self._th = threading.Thread(target=self._sample, daemon=True)
        self._th.start()

    def stop(self):
        j = self.joules() - self.j0
        self._stop.set()
        self._th.join()
        return j


# ---------------------------------------------------------------- trace shape
def arrival_times(n, peak_rate, quiet_rate, split):
    """Ramp 0->peak_rate (linear), constant peak, constant quiet. Returns [(t, phase)]."""
    n_ramp, n_peak = round(n * split[0]), round(n * split[1])
    n_quiet = n - n_ramp - n_peak
    t_ramp = 2 * n_ramp / peak_rate          # area of the ramp triangle = n_ramp
    out = [((2 * t_ramp * (k + 0.5) / peak_rate) ** 0.5, "ramp") for k in range(n_ramp)]
    out += [(t_ramp + (k + 0.5) / peak_rate, "peak") for k in range(n_peak)]
    t_quiet0 = t_ramp + n_peak / peak_rate
    out += [(t_quiet0 + (k + 0.5) / quiet_rate, "quiet") for k in range(n_quiet)]
    return out, {"ramp_end": t_ramp, "peak_end": t_quiet0, "quiet_end": t_quiet0 + n_quiet / quiet_rate}


# ---------------------------------------------------------------- one streamed request
async def stream_chat(client, base, model, messages, tools):
    body = {"model": model, "messages": messages, "stream": True, "stream_options": {"include_usage": True},
            "max_tokens": MAX_TOKENS, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": True}}
    if tools:
        body["tools"], body["tool_choice"] = tools, "auto"
    rec = {"ttft": None, "tpot": None, "prompt_tokens": None, "completion_tokens": None,
           "finish_reason": None, "error": None, "content": "", "reasoning_chars": 0, "tool_calls": []}
    calls, t_last = {}, None
    t0 = time.perf_counter()
    try:
        async with client.stream("POST", f"{base}/chat/completions", json=body, timeout=TIMEOUT_S) as r:
            if r.status_code != 200:
                rec["error"] = f"HTTP {r.status_code}: {(await r.aread())[:300]!r}"
                return rec
            async for line in r.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                ch = json.loads(data)
                if ch.get("usage"):
                    rec["prompt_tokens"] = ch["usage"].get("prompt_tokens")
                    rec["completion_tokens"] = ch["usage"].get("completion_tokens")
                for c in ch.get("choices") or []:
                    d = c.get("delta") or {}
                    piece = d.get("content") or ""
                    reason = d.get("reasoning_content") or d.get("reasoning") or ""
                    tcs = d.get("tool_calls") or []
                    if piece or reason or tcs:
                        now = time.perf_counter()
                        if rec["ttft"] is None:
                            rec["ttft"] = now - t0
                        t_last = now
                    rec["content"] += piece
                    rec["reasoning_chars"] += len(reason)
                    for tc in tcs:
                        slot = calls.setdefault(tc.get("index", 0), {"id": None, "name": "", "arguments": ""})
                        slot["id"] = tc.get("id") or slot["id"]
                        fn = tc.get("function") or {}
                        slot["name"] += fn.get("name") or ""
                        slot["arguments"] += fn.get("arguments") or ""
                    if c.get("finish_reason"):
                        rec["finish_reason"] = c["finish_reason"]
    except Exception as e:  # noqa: BLE001
        rec["error"] = repr(e)
    rec["latency"] = time.perf_counter() - t0
    rec["tool_calls"] = [calls[k] for k in sorted(calls)]
    n = rec["completion_tokens"]
    if rec["ttft"] is not None and n and n > 1:
        rec["tpot"] = (t_last - t0 - rec["ttft"]) / (n - 1)
    return rec


async def run_item(client, base, model, it, req_log, item_log, t_start, sched_t=None, phase=None):
    if sched_t is not None:
        await asyncio.sleep(max(0.0, t_start + sched_t - time.perf_counter()))
    msgs, rec = list(it["messages"]), None
    for turn in range(15):  # multi-turn tool episodes, like the grader (<= 15 turns)
        send = time.perf_counter() - t_start
        rec = await stream_chat(client, base, model, msgs, it.get("tools"))
        rec.update(id=it["id"], cls=it["cls"], turn=turn, send_t=send, phase=phase)
        req_log.append(rec)
        if rec["error"] or not rec["tool_calls"]:
            break
        msgs.append({"role": "assistant", "content": rec["content"] or None, "tool_calls": [
            {"id": c["id"] or f"call_{turn}_{i}", "type": "function",
             "function": {"name": c["name"], "arguments": c["arguments"] or "{}"}}
            for i, c in enumerate(rec["tool_calls"])]})
        for i, c in enumerate(rec["tool_calls"]):
            try:
                args = json.loads(c["arguments"] or "{}")
            except json.JSONDecodeError:
                args = {}
            msgs.append({"role": "tool", "tool_call_id": c["id"] or f"call_{turn}_{i}",
                         "content": json.dumps(call_tool(c["name"], args))})
    item_log.append({"id": it["id"], "cls": it["cls"], "grade": it["grade"], "turns": rec["turn"] + 1,
                     "answer": rec["content"], "error": rec["error"], "finish_reason": rec["finish_reason"]})


async def main_async(a):
    items = [json.loads(l) for l in open(a.workload, encoding="utf-8")]
    if a.limit:
        items = items[: a.limit]
    out = os.path.join("results", a.name)
    os.makedirs(out, exist_ok=True)
    base = a.base_url.rstrip("/")
    limits = httpx.Limits(max_connections=1000, max_keepalive_connections=1000)
    async with httpx.AsyncClient(limits=limits, timeout=TIMEOUT_S) as client:
        model = a.model or (await client.get(f"{base}/models")).json()["data"][0]["id"]
        print(f"[run] {a.mode} | model={model} | {len(items)} items -> {out}")
        warm = await stream_chat(client, base, model, [{"role": "user", "content": "Say hi."}], None)
        if warm["error"]:
            sys.exit(f"[run] warm-up failed: {warm['error']}")

        energy = Energy(os.path.join(out, "power.csv"))
        req_log, item_log = [], []
        energy.start()
        t_start = time.perf_counter()
        if a.mode == "load":
            sched, phases = arrival_times(len(items), a.peak_rate, a.quiet_rate, (a.ramp_frac, a.peak_frac))
            tasks = [run_item(client, base, model, it, req_log, item_log, t_start, t, ph)
                     for it, (t, ph) in zip(items, sched)]
            await asyncio.gather(*tasks)
        else:
            phases, sem = {}, asyncio.Semaphore(a.concurrency)

            async def guarded(it):
                async with sem:
                    await run_item(client, base, model, it, req_log, item_log, t_start)
            await asyncio.gather(*(guarded(it) for it in items))
        window = time.perf_counter() - t_start
        joules = energy.stop()

    with open(os.path.join(out, "requests.jsonl"), "w", encoding="utf-8") as f:
        for r in req_log:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(out, "items.jsonl"), "w", encoding="utf-8") as f:
        for r in sorted(item_log, key=lambda r: r["id"]):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary = {"mode": a.mode, "model": model, "workload": a.workload, "items": len(items),
               "requests": len(req_log), "window_s": window, "joules": joules,
               "avg_watts": joules / window if window else None, "power_limit_w": energy.power_limit_w(),
               "phases": phases, "note": a.note}
    json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=2)
    print(json.dumps(summary, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["load", "acc"])
    ap.add_argument("--name", required=True, help="results/<name>")
    ap.add_argument("--workload", default="data/workload.jsonl")
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--model", default=None, help="default: first id from /v1/models")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("-c", "--concurrency", type=int, default=48, help="acc mode only")
    ap.add_argument("--peak-rate", type=float, default=4.0)
    ap.add_argument("--quiet-rate", type=float, default=1.0)
    ap.add_argument("--ramp-frac", type=float, default=0.25, help="share of requests in the ramp")
    ap.add_argument("--peak-frac", type=float, default=0.55, help="share in the peak; rest is quiet")
    ap.add_argument("--note", default="", help="free text saved in summary.json (config, power cap...)")
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()

"""Estimate the GreenMind Round-1 score of a local run, using the briefing's formulas.

  python harness/score.py results/awq_mtp-s128 --baseline results/baseline
  python harness/score.py results/awq_mtp-acc  --baseline results/baseline-acc   # accuracy only

S = 100 * A * sE^0.6 * sT^0.2 * sL^0.2  (A, sE need a baseline run of the SAME workload)
This is an estimate: the hidden trace, its timing and the grader's exact grading differ.
"""
import argparse
import json
import math
import os
import re
from collections import defaultdict

FINAL_COUNTS = {"chat": 90, "rag": 60, "agentic": 60, "reasoning": 45, "thai": 30, "long": 15}
CHECKED = [c for c, n in FINAL_COUNTS.items() if n >= 20]  # per-class check applies to these

NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def final_text(t):
    return re.sub(r"<think>.*?</think>", "", t or "", flags=re.S).strip()


def is_correct(item):
    g, t = item["grade"], final_text(item["answer"])
    if item["error"] or item["finish_reason"] == "length" or not t:
        return False
    if g["type"] == "choice":
        opts = g["options"]
        m = re.search(rf"(?:answer|คำตอบ)[^\n{opts}]{{0,20}}\b([{opts}])\b", t, re.I)
        hits = [m.group(1)] if m else re.findall(rf"\b([{opts}])\b", t)
        return bool(hits) and hits[-1].upper() == g["answer"]
    boxed = re.findall(r"\\boxed\{([^}]*)\}", t)
    # Look at \boxed{} or else only the final line, and accept the answer anywhere in it
    # ("57065.22 MMK, rounded to 2 decimals"), so worked steps above it don't count.
    last = boxed[-1] if boxed else [l for l in t.splitlines() if l.strip()][-1]
    nums = [float(n.replace(",", "")) for n in NUM.findall(last)]
    if not nums:
        return False
    ans = float(g["answer"])
    if ans < 0 and re.search(r"colder|lower|less", last, re.I):  # "8 degrees colder" == -8
        nums += [-x for x in nums]
    return any(abs(x - ans) <= max(0.011, 1e-4 * abs(ans)) for x in nums)


def load(run):
    rd = lambda f: [json.loads(l) for l in open(os.path.join(run, f), encoding="utf-8")]
    return rd("items.jsonl"), rd("requests.jsonl"), json.load(open(os.path.join(run, "summary.json")))


def class_acc(items):
    tot, ok = defaultdict(int), defaultdict(int)
    for it in items:
        tot[it["cls"]] += 1
        ok[it["cls"]] += is_correct(it)
    return {c: ok[c] / tot[c] for c in tot}, dict(tot)


def on_time(r):
    if r["error"] or r["ttft"] is None or r["finish_reason"] == "length":
        return False
    if r["ttft"] > 0.5 + 0.25 * (r["prompt_tokens"] or 0) / 1000:
        return False
    return (r["completion_tokens"] or 0) <= 1 or (r["tpot"] is not None and r["tpot"] <= 0.050)


def p95(xs):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, math.ceil(0.95 * len(xs)) - 1)] if xs else float("nan")


def smoothstep(x):
    x = min(1.0, max(0.0, x))
    return 3 * x * x - 2 * x ** 3


def metrics(run):
    items, reqs, summ = load(run)
    acc, n = class_acc(items)
    ok = [on_time(r) for r in reqs]
    peak = [o for o, r in zip(ok, reqs) if r.get("phase") == "peak"]
    ttft_ratio = [r["ttft"] / (0.5 + 0.25 * (r["prompt_tokens"] or 0) / 1000) for r in reqs if r["ttft"] is not None]
    tpots = [r["tpot"] for r in reqs if r["tpot"] is not None]
    s_ttft = min(1.0, 1 / p95(ttft_ratio)) if ttft_ratio else 0.0
    s_tpot = min(1.0, 0.050 / p95(tpots)) if tpots else 0.0
    return {
        "acc": acc, "n": n, "macro_acc": sum(acc.values()) / len(acc),
        "on_time": sum(ok), "requests": len(reqs), "joules": summ["joules"],
        "eta": sum(ok) / summ["joules"] if summ["joules"] else float("nan"),
        "sT": sum(peak) / len(peak) if peak else None,
        "sL": math.sqrt(s_ttft * s_tpot), "p95_ttft_ratio": p95(ttft_ratio), "p95_tpot_ms": 1000 * p95(tpots),
        "errors": sum(1 for r in reqs if r["error"]), "cut_off": sum(1 for r in reqs if r["finish_reason"] == "length"),
        "avg_completion_tokens": sum(r["completion_tokens"] or 0 for r in reqs) / max(1, len(reqs)),
        "summary": summ,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--baseline")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    m = metrics(a.run)
    b = metrics(a.baseline) if a.baseline else None

    print(f"run: {a.run}   ({m['summary']['mode']}, power limit {m['summary']['power_limit_w']} W)")
    print(f"{'class':<10}{'n':>4}{'acc':>8}" + (f"{'base':>8}{'ratio':>8}" if b else ""))
    for c in FINAL_COUNTS:
        if c not in m["acc"]:
            continue
        line = f"{c:<10}{m['n'][c]:>4}{m['acc'][c]:>8.3f}"
        if b and c in b["acc"]:
            ratio = m["acc"][c] / b["acc"][c] if b["acc"][c] else float("nan")
            line += f"{b['acc'][c]:>8.3f}{ratio:>8.3f}" + ("  <-- below 0.75 of baseline" if ratio < 0.75 else "")
        print(line)
    out_tokens = m["avg_completion_tokens"] * m["requests"]
    print(f"on-time {m['on_time']}/{m['requests']}  errors {m['errors']}  cut-off {m['cut_off']}  "
          f"joules {m['joules']:.0f}  avg W {m['summary']['avg_watts'] or 0:.0f}  "
          f"avg out tok {m['avg_completion_tokens']:.0f}  "
          # output length varies run to run (+-8%); compare configs per token, not just per answer
          f"mJ/token {1000 * m['joules'] / out_tokens if out_tokens else float('nan'):.1f}")
    print(f"p95 TTFT/SLO {m['p95_ttft_ratio']:.2f}  p95 TPOT {m['p95_tpot_ms']:.1f} ms  "
          f"sL {m['sL']:.3f}  sT {m['sT'] if m['sT'] is None else round(m['sT'], 3)}")

    out = {"sL": m["sL"], "sT": m["sT"]}
    if b:
        R = m["macro_acc"] / b["macro_acc"]
        cf = min((smoothstep((m["acc"][c] / b["acc"][c] - 0.5) / 0.25)
                  for c in CHECKED if c in m["acc"] and b["acc"].get(c)), default=1.0)
        A = max(0.05, smoothstep((R - 0.85) / 0.13) * cf)
        ratio = m["eta"] / b["eta"] if b["eta"] else float("nan")
        sE = min(1.0, max(0.0, math.log(ratio) / math.log(3))) if ratio > 0 else 0.0
        sT = m["sT"] if m["sT"] is not None else 1.0
        S = 100 * A * sE ** 0.6 * sT ** 0.2 * m["sL"] ** 0.2
        out.update(R=R, class_factor=cf, A=A, eff_x=ratio, sE=sE, S=S)
        print(f"R {R:.3f}  class_factor {cf:.3f}  A {A:.3f}  |  efficiency {ratio:.2f}x baseline  sE {sE:.3f}")
        print(f"==> estimated S = {S:.1f}" + ("" if m["summary"]["mode"] == "load" else "   (acc-mode run: energy terms not grader-like)"))
    if a.json:
        print(json.dumps(out))


if __name__ == "__main__":
    main()

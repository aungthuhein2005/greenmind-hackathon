"""Median and worst case per config over repeated runs named <config>-r<N>.

  python harness/aggregate.py rep-s36 rep-s40 rep-s44
"""
import glob
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score import metrics  # noqa: E402

print(f"{'config':<10}{'runs':>5}{'on-time med/min':>17}{'mJ/tok med/max':>16}{'J/req med':>10}"
      f"{'sT med/min':>12}{'TPOT95 med/max':>16}{'TTFT/SLO med':>13}")
for cfg in sys.argv[1:]:
    ms = [metrics(d) for d in sorted(glob.glob(f"results/{cfg}-r*")) if os.path.isfile(f"{d}/summary.json")]
    if not ms:
        continue
    on = [m["on_time"] for m in ms]
    mj = [1000 * m["joules"] / (m["avg_completion_tokens"] * m["requests"]) for m in ms]
    jr = [m["joules"] / m["requests"] for m in ms]
    st = [m["sT"] or 0 for m in ms]
    tp = [m["p95_tpot_ms"] for m in ms]
    tt = [m["p95_ttft_ratio"] for m in ms]
    print(f"{cfg:<10}{len(ms):>5}{statistics.median(on):>11.0f} / {min(on):<3}{statistics.median(mj):>9.1f} / {max(mj):<5.1f}"
          f"{statistics.median(jr):>9.0f}{statistics.median(st):>8.2f} / {min(st):.2f}"
          f"{statistics.median(tp):>9.1f} / {max(tp):<5.1f}{statistics.median(tt):>10.0f}")

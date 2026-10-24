#!/usr/bin/env bash
# Sweep power limits against an already-running server (no restart needed:
# the limit can change between runs). One local load run per setting.
#   ./serve/sweep_power.sh awq_mtp-s128 "400 300 250 220 200"
set -euo pipefail
TAG="${1:?usage: sweep_power.sh <config-tag> \"<watts list>\"}"
LIMITS="${2:-400 300 250 220 200}"

for PL in $LIMITS; do
  ./serve/power.sh "$PL" >/dev/null
  sleep 5
  python harness/run.py load --name "$TAG-pl$PL"
  python harness/score.py "results/$TAG-pl$PL" --baseline results/baseline | tee "results/$TAG-pl$PL/score.txt"
done

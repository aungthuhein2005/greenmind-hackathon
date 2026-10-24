#!/usr/bin/env bash
# Set a constant GPU power limit (and optionally locked SM clocks).
# Rules allow this only if it is held for the WHOLE run -> set it before submitting
# and leave it alone until the grader finishes.
#   ./serve/power.sh 250            # power limit 250 W
#   ./serve/power.sh 250 1410       # + lock SM clocks at <=1410 MHz
#   ./serve/power.sh reset          # back to defaults
set -euo pipefail

if [[ "${1:-}" == "reset" ]]; then
  sudo nvidia-smi -rgc
  sudo nvidia-smi -pl "$(nvidia-smi --query-gpu=power.default_limit --format=csv,noheader,nounits | cut -d. -f1)"
else
  PL="${1:?usage: power.sh <watts> [max_sm_mhz] | reset}"
  sudo nvidia-smi -pm 1
  sudo nvidia-smi -pl "$PL"
  if [[ -n "${2:-}" ]]; then sudo nvidia-smi -lgc "0,$2"; fi
fi

nvidia-smi --query-gpu=name,power.limit,power.min_limit,power.max_limit,clocks.max.sm,power.draw --format=csv

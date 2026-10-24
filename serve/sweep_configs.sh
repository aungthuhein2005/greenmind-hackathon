#!/usr/bin/env bash
# Run several server configs back to back: restart vLLM, wait for ready, load test, score.
# Each line of the config file is: <tag> <profile> [ENV=VAL ...]
#   ./serve/sweep_configs.sh serve/sweeps/next.txt
# Run inside tmux; results land in results/<tag>/ and a one-line summary in results/sweep.tsv.
set -uo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
CONFIGS="${1:?usage: sweep_configs.sh <config-file>}"
WORKLOAD="${WORKLOAD:-data/workload.jsonl}"   # data/val.jsonl ~ grader Validation load

wait_ready() {
  for _ in $(seq 1 90); do
    curl -sf localhost:8000/v1/models >/dev/null && return 0
    tmux has-session -t serve 2>/dev/null || return 1
    sleep 10
  done
  return 1
}

while read -r TAG PROFILE ENVS; do
  [[ -z "${TAG:-}" || "$TAG" == \#* ]] && continue
  echo "=== $TAG ($PROFILE $ENVS)"
  tmux kill-session -t serve 2>/dev/null; sleep 8
  tmux new-session -d -s serve "cd $PWD && source .venv/bin/activate && env $ENVS ./serve/serve.sh $PROFILE 2>&1 | tee results/_launch/$TAG.log"
  if ! wait_ready; then
    echo "$TAG	SERVER_FAILED	$ENVS" | tee -a results/sweep.tsv
    continue
  fi
  python harness/run.py load --name "$TAG" --workload "$WORKLOAD" --note "$PROFILE $ENVS" > "results/$TAG.run.log" 2>&1
  python harness/score.py "results/$TAG" --baseline results/baseline > "results/$TAG/score.txt" 2>&1
  S=$(grep -oE "estimated S = [0-9.]+" "results/$TAG/score.txt" | awk '{print $4}')
  OT=$(grep -oE "on-time [0-9]+/[0-9]+" "results/$TAG/score.txt" | awk '{print $2}')
  EFF=$(grep -oE "efficiency [0-9.]+x" "results/$TAG/score.txt" | awk '{print $2}')
  TPOT=$(grep -oE "p95 TPOT [0-9.]+" "results/$TAG/score.txt" | awk '{print $3}')
  A=$(grep -oE " A [0-9.]+" "results/$TAG/score.txt" | awk '{print $2}')
  J=$(grep -oE "joules [0-9]+" "results/$TAG/score.txt" | awk '{print $2}')
  JPO=$(awk -v j="$J" -v o="${OT%%/*}" 'BEGIN{ if (o>0) printf "%.0f", j/o; else print "inf" }')
  echo "$TAG	S=$S	A=$A	eff=$EFF	ontime=$OT	J/ontime=$JPO	tpot95=$TPOT	$ENVS" | tee -a results/sweep.tsv
done < "$CONFIGS"

#!/usr/bin/env bash
# Power-cap sweep on the Final-40.0 config, done right this time: the server restarts before every
# run (cold prefix cache, unlike sweep 7) and we compare energy per output token as well as on-time.
#   tmux new-session -d -s sweep "bash serve/sweeps/power_cold.sh"
cd ~/greenmind && source .venv/bin/activate
CFG="PREFIX_CACHE=1 MAX_SEQS=40 SCHED=newest"

for PL in 350 300 400; do   # 400 last: leaves the grader endpoint on the best-known setting
  TAG="s40-cold-pl$PL"
  tmux kill-session -t serve 2>/dev/null; sleep 8
  ./serve/power.sh $PL > /dev/null
  tmux new-session -d -s serve "cd $PWD && source .venv/bin/activate && env $CFG ./serve/serve.sh awq_mtp 2>&1 | tee results/_launch/$TAG.log"
  for i in $(seq 1 90); do curl -sf localhost:8000/v1/models >/dev/null && break; sleep 10; done
  python harness/run.py load --name "$TAG" --note "$CFG ${PL}W cold" > "results/$TAG.run.log" 2>&1
  python harness/score.py "results/$TAG" --baseline results/baseline > "results/$TAG/score.txt" 2>&1
  echo "$TAG $(grep -oE "on-time [0-9]+/[0-9]+|joules [0-9]+|avg W [0-9]+|mJ/token [0-9.]+|p95 TPOT [0-9.]+ ms|sT [0-9.]+" results/$TAG/score.txt | tr "\n" " ")" | tee -a results/power_cold.txt
done

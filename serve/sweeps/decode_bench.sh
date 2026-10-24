#!/usr/bin/env bash
# Pure-decode capacity for MTP on/off at high max-num-seqs, then restore the Final-40.0 server.
cd ~/greenmind && source .venv/bin/activate
./serve/power.sh 400 > /dev/null
start() {  # $1 = tag, rest = env + profile
  local TAG=$1; shift
  tmux kill-session -t serve 2>/dev/null; sleep 8
  tmux new-session -d -s serve "cd $PWD && source .venv/bin/activate && env $* 2>&1 | tee results/_launch/$TAG.log"
  for i in $(seq 1 90); do curl -sf localhost:8000/v1/models >/dev/null && return 0; sleep 10; done
  return 1
}
start bench-mtp   PREFIX_CACHE=1 MAX_SEQS=256 ./serve/serve.sh awq_mtp && python harness/decode_bench.py --tag mtp   | tee -a results/decode_bench.txt
start bench-nomtp PREFIX_CACHE=1 MAX_SEQS=256 ./serve/serve.sh awq     && python harness/decode_bench.py --tag nomtp | tee -a results/decode_bench.txt
grep -E "Capturing CUDA graphs \(FULL\)" results/_launch/bench-mtp.log results/_launch/bench-nomtp.log | grep -oE "^[^:]+|[0-9]+/[0-9]+ \[" | paste - - | tee -a results/decode_bench.txt
start grader-best-40c PREFIX_CACHE=1 MAX_SEQS=40 SCHED=newest ./serve/serve.sh awq_mtp

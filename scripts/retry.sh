#!/usr/bin/env bash
# Run a resumable GPU script and restart it after a crash (e.g. a CUDA "unspecified launch failure"
# when another app competes for the GPU). The script must be resumable (run_xai_bench.py and
# eval_xai_verification.py are).
# Usage: bash scripts/retry.sh <log file> <max tries> python scripts/xxx.py args...
log=$1; tries=$2; shift 2
for i in $(seq 1 "$tries"); do
  echo "=== attempt $i/$tries $(date) ===" >> "$log"
  PYTHONUNBUFFERED=1 "$@" >> "$log" 2>&1 && { echo "=== finished $(date) ===" >> "$log"; exit 0; }
  echo "=== crashed (exit $?), restarting in 20 s ===" >> "$log"
  sleep 20
done
echo "=== gave up after $tries attempts ===" >> "$log"
exit 1

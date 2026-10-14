#!/usr/bin/env bash
# Runs all remaining A5/A6 compute unattended (no internet needed), one GPU job at a time.
# Each step is resumable and is retried after crashes (e.g. a CUDA error after the laptop wakes up).
# Log: results/run_rest_offline.log
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python
LOG=results/run_rest_offline.log
echo "=== start $(date) ===" >> $LOG

# 1. finish the final test benchmark (waits for the running instance, then re-runs to resume if it died)
while ! grep -q "=== finished\|gave up" results/bench_test_final.log && tasklist 2>/dev/null | grep -qi python; do
  sleep 60
done
bash scripts/retry.sh results/bench_test_final.log 15 $PY scripts/run_xai_bench.py --split test --tag final \
  --stability --sanity --n-diag 150 --n-expensive 200 --methods eigencam eigencam_p5 gradcam gradcampp \
  gradcampp_p5 gradcampp_p4 gradcampp_p3 layercam layercam_p5 occlusion drise random boxprior
echo "bench done $(date)" >> $LOG

# 2. tables + plots (CPU)
PYTHONIOENCODING=utf-8 $PY scripts/analyze_bench.py results/xai_bench/test_final >> $LOG 2>&1
PYTHONIOENCODING=utf-8 $PY scripts/analyze_bench.py results/xai_bench/valid_dev >> $LOG 2>&1
echo "analysis done $(date)" >> $LOG

# 3. qualitative figures + matched pairs (retry on GPU faults)
for i in 1 2 3 4 5; do
  PYTHONUNBUFFERED=1 $PY scripts/make_figures.py --split test --pool 400 --per-kind 4 --pairs 8 >> $LOG 2>&1 && break
  echo "figures crashed, retry $i" >> $LOG; sleep 20
done
echo "figures done $(date)" >> $LOG

# 4. explanation-based verification (resumable)
bash scripts/retry.sh results/xai_verification.log 15 $PY scripts/eval_xai_verification.py --conf 0.1
echo "=== ALL DONE $(date) ===" >> $LOG

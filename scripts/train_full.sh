#!/usr/bin/env bash
# Full-scale training. LONG-RUNNING: ~10 h on CPU at ~10 ms/step.
# Use --device cuda and/or a GPU host if available (not guaranteed faster at
# batch size 64).
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-micromamba run -n test-py312 python}"
DEVICE="${DEVICE:-cpu}"

$PY -m rlmalloc.train \
  --dist lognormal_train \
  --episodes 10000 \
  --max-steps 2000 \
  --seed 0 \
  --device "$DEVICE" \
  --out results/checkpoints/agent \
  --log results/metrics/train_log.csv

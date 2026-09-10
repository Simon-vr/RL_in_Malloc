#!/usr/bin/env bash
# Bounded smoke run (~5-10 min CPU). Safe for CI / quick verification.
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-micromamba run -n test-py312 python}"

$PY -m rlmalloc.train \
  --dist lognormal_train \
  --episodes 300 \
  --max-steps 200 \
  --learn-every 2 \
  --seed 0 \
  --device cpu \
  --out results/checkpoints/agent_quick \
  --log results/metrics/train_log_quick.csv

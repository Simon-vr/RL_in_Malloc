#!/usr/bin/env bash
# Evaluate 4 policies x 4 distributions. Requires a trained checkpoint.
# For a bounded preview add:  --rounds 50
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-micromamba run -n test-py312 python}"

$PY -m rlmalloc.evaluate \
  --checkpoint results/checkpoints/agent \
  --seed 0 \
  --device cpu \
  --rounds 1000 \
  --requests 200 \
  --release-rate 0.3 \
  --dists lognormal_train lognormal_large uniform bimodal \
  --out-metrics results/metrics \
  --out-tables results/tables \
  --figures results/figures

#!/bin/bash
# Train all 5 ratio configurations x 5 seeds reported in the paper.
# Usage: bash scripts/run_all.sh [DEVICE_ID]

DEVICE=${1:-0}
SEEDS=(42 1005 1111 2026 9999)

for s in "${SEEDS[@]}"; do
    # baseline: H_sig = 0 (softmax only)
    python run.py -seed "$s" -d "$DEVICE" -m v

    # MoFA: H_sig in {3, 6, 9, 12}
    for n in 3 6 9 12; do
        python run.py -seed "$s" -d "$DEVICE" -m h -n_sig "$n"
    done
done

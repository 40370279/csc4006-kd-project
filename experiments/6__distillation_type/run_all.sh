#!/bin/bash
set -euo pipefail

PROJECT_DIR="/users/40370279/csc4006/code"
EXP_DIR="$PROJECT_DIR/experiments/6__distillation_type"

cd "$PROJECT_DIR"

echo "===== SUBMITTING EXPERIMENT 6: DISTILLATION TYPE ====="

for SCRIPT in baseline_ce_only.slurm soft_targets_only.slurm feature_only.slurm soft_targets_plus_features.slurm
do
    SBATCH_OUTPUT=$(sbatch "$EXP_DIR/$SCRIPT")
    JOB_ID=$(echo "$SBATCH_OUTPUT" | awk '{print $4}')

    if [[ -z "${JOB_ID:-}" ]]; then
        echo "Failed to submit $SCRIPT"
        echo "sbatch output: $SBATCH_OUTPUT"
        exit 1
    fi

    echo "Submitted $SCRIPT -> Job $JOB_ID"
done
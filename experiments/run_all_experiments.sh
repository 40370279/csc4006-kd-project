#!/bin/bash
set -euo pipefail

EXPERIMENTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$EXPERIMENTS_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

mkdir -p logs

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/run_all_experiments_${TIMESTAMP}.log"

exec > >(tee -a "$LOG_FILE") 2>&1

echo "===== RUNNING ALL EXPERIMENTS ====="
echo "Time: $(date)"
echo

for EXP_DIR in experiments/*/
do
    if [[ -f "${EXP_DIR}run_all.sh" ]]; then
        echo "--------------------------------------"
        echo "Running: ${EXP_DIR}run_all.sh"
        echo "--------------------------------------"

        bash "${EXP_DIR}run_all.sh"

        echo
        echo "Finished: ${EXP_DIR}"
        echo
    else
        echo "Skipping ${EXP_DIR} (no run_all.sh)"
    fi
done

echo "===== ALL EXPERIMENTS COMPLETE ====="
echo "Finished at: $(date)"
echo "Log saved to: $LOG_FILE"
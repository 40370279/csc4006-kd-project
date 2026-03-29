#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(pwd)"
EXP_DIR="$PROJECT_ROOT/experiments"

echo "===== RUNNING ALL EXPERIMENTS ====="
echo "Start time: $(date)"
echo "Project root: $PROJECT_ROOT"
echo

EXPERIMENTS=(
  "1__kd_vs_baseline"
  "2__temperature_sweep"
  "3__alpha_sweep"
  "4__student_capacity_sweep"
  "5__teacher_model_comparison"
  "6__distillation_component"
  "7__augmentation_ablation"
)

for EXP in "${EXPERIMENTS[@]}"
do
  SCRIPT_PATH="$EXP_DIR/$EXP/run_all.sh"

  echo "----------------------------------------"
  echo "Running experiment: $EXP"
  echo "Script: $SCRIPT_PATH"
  echo

  if [[ ! -f "$SCRIPT_PATH" ]]; then
    echo "ERROR: $SCRIPT_PATH not found"
    exit 1
  fi

  chmod +x "$SCRIPT_PATH"
  bash "$SCRIPT_PATH"

  echo
  echo "Finished experiment: $EXP"
  echo "----------------------------------------"
  echo
done

echo "===== ALL EXPERIMENTS SUBMITTED ====="
echo "End time: $(date)"
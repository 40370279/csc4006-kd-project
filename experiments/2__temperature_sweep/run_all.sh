#!/bin/bash
set -euo pipefail

ROOT_DIR="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"

echo "===== RUNNING EXPERIMENT 2: TEMPERATURE SWEEP ====="
echo "Directory : $ROOT_DIR"
echo "Host      : $(hostname)"
echo "Start     : $(date)"
echo

FILES=(
  "experiments/2__temperature_sweep/run_T1.slurm"
  "experiments/2__temperature_sweep/run_T2.slurm"
  "experiments/2__temperature_sweep/run_T4.slurm"
  "experiments/2__temperature_sweep/run_T8.slurm"
  "experiments/2__temperature_sweep/run_T16.slurm"
  "experiments/2__temperature_sweep/run_T32.slurm"
)

JOB_IDS=()

cd "$ROOT_DIR"

for FILE in "${FILES[@]}"
do
  if [[ ! -f "$FILE" ]]; then
    echo "ERROR: Missing file $FILE"
    exit 1
  fi

  echo "Submitting $FILE ..."
  SBATCH_OUTPUT=$(sbatch "$FILE")
  echo "$SBATCH_OUTPUT"

  JOB_ID=$(echo "$SBATCH_OUTPUT" | awk '{print $NF}')
  JOB_IDS+=("$JOB_ID")
done

echo
echo "===== SUBMISSION SUMMARY ====="
for i in "${!FILES[@]}"
do
  printf "%-45s -> %s\n" "${FILES[$i]}" "${JOB_IDS[$i]}"
done

echo
echo "Submitted at: $(date)"
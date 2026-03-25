#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "===== RUNNING EXPERIMENT 3: ALPHA SWEEP ====="
echo "Directory : $SCRIPT_DIR"
echo "Host      : $(hostname)"
echo "Start     : $(date)"
echo

FILES=(
  "run_A01.slurm"
  "run_A03.slurm"
  "run_A05.slurm"
  "run_A07.slurm"
  "run_A09.slurm"
)

JOB_IDS=()

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
  printf "%-15s -> %s\n" "${FILES[$i]}" "${JOB_IDS[$i]}"
done

echo
echo "Submitted at: $(date)"
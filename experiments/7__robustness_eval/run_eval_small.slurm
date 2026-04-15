#!/bin/bash
#SBATCH --job-name=exp7_robust_small
#SBATCH --partition=k2-gpu-v100
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --output=/users/40370279/csc4006/code/experiments/7__robustness_eval/logs/robust_small_%j.out

set -euo pipefail

module purge
module load python3/3.10.5/gcc-9.3.0

PROJECT_DIR="/users/40370279/csc4006/code"
EXP_DIR="$PROJECT_DIR/experiments/7__robustness_eval"
LOG_DIR="$EXP_DIR/logs"
TMP_LOG_DIR="$LOG_DIR/tmp"
RESULTS_DIR="$EXP_DIR/results"

cd "$PROJECT_DIR"

export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

mkdir -p "$LOG_DIR" "$TMP_LOG_DIR" "$RESULTS_DIR"

echo "===== EXPERIMENT 7: ROBUSTNESS EVALUATION (SMALL) ====="
echo "Baseline checkpoints source: experiments/1__kd_vs_baseline/checkpoints/"
echo "KD checkpoints source      : experiments/1__kd_vs_baseline/checkpoints/"
echo "Seeds: 42, 123, 999"
echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Host: $(hostname)"
echo "Start: $(date)"
echo "Working directory: $(pwd)"
echo

SEEDS=(42 123 999)

CLEAN_MACRO_ADV_VALUES=()
NOISE05_MACRO_ADV_VALUES=()
NOISE10_MACRO_ADV_VALUES=()
AMP10_MACRO_ADV_VALUES=()
AMP20_MACRO_ADV_VALUES=()
LEAD1_MACRO_ADV_VALUES=()
LEAD2_MACRO_ADV_VALUES=()
TMASK05_MACRO_ADV_VALUES=()
TMASK10_MACRO_ADV_VALUES=()

for SEED in "${SEEDS[@]}"
do
  echo "----- SEED $SEED -----"

  SEED_LOG="$TMP_LOG_DIR/robust_small_seed_${SEED}_${SLURM_JOB_ID:-manual}.log"

  BASELINE_CKPT="$PROJECT_DIR/experiments/1__kd_vs_baseline/checkpoints/student_baseline_seed${SEED}.pt"
  KD_CKPT="$PROJECT_DIR/experiments/1__kd_vs_baseline/checkpoints/student_kd_seed${SEED}.pt"
  OUTPUT_JSON="$RESULTS_DIR/robustness_small_seed${SEED}.json"

  if [[ ! -f "$BASELINE_CKPT" ]]; then
    echo "ERROR: Missing baseline checkpoint: $BASELINE_CKPT"
    exit 1
  fi

  if [[ ! -f "$KD_CKPT" ]]; then
    echo "ERROR: Missing KD checkpoint: $KD_CKPT"
    exit 1
  fi

  python3 -u experiments/7__robustness_eval/evaluate_robustness.py \
    --baseline_ckpt "$BASELINE_CKPT" \
    --kd_ckpt "$KD_CKPT" \
    --batch_size 128 \
    --num_workers 2 \
    --seed "$SEED" \
    --output_json "$OUTPUT_JSON" \
    2>&1 | tee "$SEED_LOG"

  CLEAN_MACRO_ADV=$(python3 - <<EOF
import json
with open("$OUTPUT_JSON", "r", encoding="utf-8") as f:
    data = json.load(f)
for row in data["settings"]:
    if row["corruption"] == "clean":
        print(row["kd_advantage"]["macro_f1_advantage"])
        break
EOF
)

  NOISE05_MACRO_ADV=$(python3 - <<EOF
import json
with open("$OUTPUT_JSON", "r", encoding="utf-8") as f:
    data = json.load(f)
for row in data["settings"]:
    if row["corruption"] == "gaussian_noise" and abs(row["severity"] - 0.05) < 1e-9:
        print(row["kd_advantage"]["macro_f1_advantage"])
        break
EOF
)

  NOISE10_MACRO_ADV=$(python3 - <<EOF
import json
with open("$OUTPUT_JSON", "r", encoding="utf-8") as f:
    data = json.load(f)
for row in data["settings"]:
    if row["corruption"] == "gaussian_noise" and abs(row["severity"] - 0.10) < 1e-9:
        print(row["kd_advantage"]["macro_f1_advantage"])
        break
EOF
)

  AMP10_MACRO_ADV=$(python3 - <<EOF
import json
with open("$OUTPUT_JSON", "r", encoding="utf-8") as f:
    data = json.load(f)
for row in data["settings"]:
    if row["corruption"] == "amplitude_scale" and abs(row["severity"] - 0.10) < 1e-9:
        print(row["kd_advantage"]["macro_f1_advantage"])
        break
EOF
)

  AMP20_MACRO_ADV=$(python3 - <<EOF
import json
with open("$OUTPUT_JSON", "r", encoding="utf-8") as f:
    data = json.load(f)
for row in data["settings"]:
    if row["corruption"] == "amplitude_scale" and abs(row["severity"] - 0.20) < 1e-9:
        print(row["kd_advantage"]["macro_f1_advantage"])
        break
EOF
)

  LEAD1_MACRO_ADV=$(python3 - <<EOF
import json
with open("$OUTPUT_JSON", "r", encoding="utf-8") as f:
    data = json.load(f)
for row in data["settings"]:
    if row["corruption"] == "lead_dropout" and abs(row["severity"] - (1.0/12.0)) < 1e-9:
        print(row["kd_advantage"]["macro_f1_advantage"])
        break
EOF
)

  LEAD2_MACRO_ADV=$(python3 - <<EOF
import json
with open("$OUTPUT_JSON", "r", encoding="utf-8") as f:
    data = json.load(f)
for row in data["settings"]:
    if row["corruption"] == "lead_dropout" and abs(row["severity"] - (2.0/12.0)) < 1e-9:
        print(row["kd_advantage"]["macro_f1_advantage"])
        break
EOF
)

  TMASK05_MACRO_ADV=$(python3 - <<EOF
import json
with open("$OUTPUT_JSON", "r", encoding="utf-8") as f:
    data = json.load(f)
for row in data["settings"]:
    if row["corruption"] == "time_mask" and abs(row["severity"] - 0.05) < 1e-9:
        print(row["kd_advantage"]["macro_f1_advantage"])
        break
EOF
)

  TMASK10_MACRO_ADV=$(python3 - <<EOF
import json
with open("$OUTPUT_JSON", "r", encoding="utf-8") as f:
    data = json.load(f)
for row in data["settings"]:
    if row["corruption"] == "time_mask" and abs(row["severity"] - 0.10) < 1e-9:
        print(row["kd_advantage"]["macro_f1_advantage"])
        break
EOF
)

  if [[ -z "${CLEAN_MACRO_ADV:-}" || -z "${NOISE05_MACRO_ADV:-}" || -z "${NOISE10_MACRO_ADV:-}" || -z "${AMP10_MACRO_ADV:-}" || -z "${AMP20_MACRO_ADV:-}" || -z "${LEAD1_MACRO_ADV:-}" || -z "${LEAD2_MACRO_ADV:-}" || -z "${TMASK05_MACRO_ADV:-}" || -z "${TMASK10_MACRO_ADV:-}" ]]; then
    echo "ERROR: Failed to parse robustness metrics for seed $SEED"
    echo "Check log: $SEED_LOG"
    echo "Check json: $OUTPUT_JSON"
    exit 1
  fi

  CLEAN_MACRO_ADV_VALUES+=("$CLEAN_MACRO_ADV")
  NOISE05_MACRO_ADV_VALUES+=("$NOISE05_MACRO_ADV")
  NOISE10_MACRO_ADV_VALUES+=("$NOISE10_MACRO_ADV")
  AMP10_MACRO_ADV_VALUES+=("$AMP10_MACRO_ADV")
  AMP20_MACRO_ADV_VALUES+=("$AMP20_MACRO_ADV")
  LEAD1_MACRO_ADV_VALUES+=("$LEAD1_MACRO_ADV")
  LEAD2_MACRO_ADV_VALUES+=("$LEAD2_MACRO_ADV")
  TMASK05_MACRO_ADV_VALUES+=("$TMASK05_MACRO_ADV")
  TMASK10_MACRO_ADV_VALUES+=("$TMASK10_MACRO_ADV")

  echo "Parsed robustness macro-F1 advantages for seed $SEED:"
  echo "  Clean                 = $CLEAN_MACRO_ADV"
  echo "  Gaussian noise 0.05   = $NOISE05_MACRO_ADV"
  echo "  Gaussian noise 0.10   = $NOISE10_MACRO_ADV"
  echo "  Amplitude scale 0.10  = $AMP10_MACRO_ADV"
  echo "  Amplitude scale 0.20  = $AMP20_MACRO_ADV"
  echo "  Lead dropout 1 lead   = $LEAD1_MACRO_ADV"
  echo "  Lead dropout 2 leads  = $LEAD2_MACRO_ADV"
  echo "  Time mask 0.05        = $TMASK05_MACRO_ADV"
  echo "  Time mask 0.10        = $TMASK10_MACRO_ADV"
  echo
done

CLEAN_STR="${CLEAN_MACRO_ADV_VALUES[*]}"
NOISE05_STR="${NOISE05_MACRO_ADV_VALUES[*]}"
NOISE10_STR="${NOISE10_MACRO_ADV_VALUES[*]}"
AMP10_STR="${AMP10_MACRO_ADV_VALUES[*]}"
AMP20_STR="${AMP20_MACRO_ADV_VALUES[*]}"
LEAD1_STR="${LEAD1_MACRO_ADV_VALUES[*]}"
LEAD2_STR="${LEAD2_MACRO_ADV_VALUES[*]}"
TMASK05_STR="${TMASK05_MACRO_ADV_VALUES[*]}"
TMASK10_STR="${TMASK10_MACRO_ADV_VALUES[*]}"

echo "===== SUMMARY: ROBUSTNESS EVALUATION (SMALL) ====="

python3 - <<EOF
import numpy as np

clean = np.array([float(x) for x in """$CLEAN_STR""".split()])
noise05 = np.array([float(x) for x in """$NOISE05_STR""".split()])
noise10 = np.array([float(x) for x in """$NOISE10_STR""".split()])
amp10 = np.array([float(x) for x in """$AMP10_STR""".split()])
amp20 = np.array([float(x) for x in """$AMP20_STR""".split()])
lead1 = np.array([float(x) for x in """$LEAD1_STR""".split()])
lead2 = np.array([float(x) for x in """$LEAD2_STR""".split()])
tmask05 = np.array([float(x) for x in """$TMASK05_STR""".split()])
tmask10 = np.array([float(x) for x in """$TMASK10_STR""".split()])

print("Experiment   : ROBUSTNESS EVALUATION (SMALL)")
print("Seeds        :", " ".join(map(str, [42, 123, 999])))
print(f"Clean Macro-F1 advantage               : {clean.mean():.4f} ± {clean.std():.4f}")
print(f"Gaussian noise 0.05 Macro-F1 advantage : {noise05.mean():.4f} ± {noise05.std():.4f}")
print(f"Gaussian noise 0.10 Macro-F1 advantage : {noise10.mean():.4f} ± {noise10.std():.4f}")
print(f"Amplitude scale 0.10 advantage         : {amp10.mean():.4f} ± {amp10.std():.4f}")
print(f"Amplitude scale 0.20 advantage         : {amp20.mean():.4f} ± {amp20.std():.4f}")
print(f"Lead dropout 1 lead advantage          : {lead1.mean():.4f} ± {lead1.std():.4f}")
print(f"Lead dropout 2 leads advantage         : {lead2.mean():.4f} ± {lead2.std():.4f}")
print(f"Time mask 0.05 advantage               : {tmask05.mean():.4f} ± {tmask05.std():.4f}")
print(f"Time mask 0.10 advantage               : {tmask10.mean():.4f} ± {tmask10.std():.4f}")
EOF

echo
echo "Finished: $(date)"
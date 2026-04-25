![CI](https://github.com/lewismcgrogan/CSC4006-KD-Project/actions/workflows/ci.yml/badge.svg)

# ECG Knowledge Distillation for Efficient PTB-XL Classification

This repository contains a reproducible research software framework for investigating **knowledge distillation (KD)** for efficient ECG classification on the **PTB-XL** dataset.

The project studies whether a compact student model can approach the performance of a higher-capacity teacher model while remaining significantly smaller and faster at inference.

## Project objectives

The software is designed to support the following goals:

- Train a high-capacity **teacher** ECG classifier
- Train a lightweight **baseline student** classifier
- Train a lightweight **knowledge-distilled student**
- Compare teacher, baseline, and KD models under controlled settings
- Evaluate both **predictive performance** and **deployment-oriented efficiency**
- Support reproducible ablation studies and robustness experiments
- Provide a maintainable research software artefact that can be installed, executed, tested, and extended by another researcher or research software engineer

## Repository structure

```text
code/
├── .github/workflows/       # GitHub Actions CI workflow
├── src/
│   ├── data/                # Dataset wrapper and ECG augmentation
│   ├── models/              # Teacher and student model implementations
│   └── utils/               # Metrics, model statistics, ROC plotting utilities
├── scripts/                 # Main Python entry points
├── slurm/                   # SLURM job scripts for Kelvin2
├── experiments/             # Experiment-specific runs and logs
├── tests/                   # Lightweight automated tests
├── docs/                    # Additional QA notes and known issues
├── data/ptbxl/              # Raw PTB-XL dataset location
├── processed/               # Preprocessed dataset outputs
├── checkpoints/             # Trained model checkpoints
├── logs/                    # Log files
├── results/                 # Generated figures and result outputs
├── README.md
├── INSTALL.md
├── REPLICATION_GUIDE.md
├── TESTING.md
├── LICENSE
└── requirements.txt
```

## Execution model

The project can be used in two main ways:

- **Local execution**, by running the main Python scripts directly
- **Kelvin2 / SLURM execution**, by submitting the provided batch scripts

In practice, local execution is mainly intended for setup, preprocessing, and functional verification of the core training workflows. Full grouped experiment reproduction is primarily intended for Kelvin2, because the experiment suite is organised around batch-oriented experiment scripts and may be impractically slow to run fully on local hardware.

### Working directory requirement

Most training, experiment, plotting, and SLURM workflows are designed to be run from the repository root, for example:

```bash
cd /users/40370279/csc4006/code
```

This is important because many scripts save logs, checkpoints, processed data, and generated figures using project-root-relative paths such as:

- `logs/`
- `checkpoints/`
- `processed/`
- `results/`
- `experiments/.../logs/`
- `experiments/.../checkpoints/`

Running scripts from a different working directory may cause outputs to be written to unexpected locations or may prevent checkpoint/data paths from resolving correctly. The provided SLURM scripts therefore change into the project root before execution.

## Main components

### Preprocessing

**Script:** `scripts/preprocess_ptbxl.py`

This script:

- Loads PTB-XL metadata and waveform files
- Filters to **500 Hz** recordings
- Maps ECG records to **5 diagnostic superclasses**
- Keeps only records with a **single unambiguous superclass**
- Crops or pads each ECG to **12 × 5000**
- Applies per-lead normalisation
- Creates train, validation, and test splits using PTB-XL folds

**Output:**

- `processed/ptbxl_500hz_10s.npz`

### Teacher model

**Script:** `scripts/train_teacher.py`

The teacher is a higher-capacity CNN designed to learn strong ECG representations. It is used both as a standalone classifier and as the supervision source for knowledge distillation.

### Baseline student model

**Script:** `scripts/train_student_baseline.py`

The baseline student is a lightweight residual CNN trained using standard supervised learning only.

### KD student model

**Script:** `scripts/train_student_kd.py`

The KD student is trained using a combined objective including:

- Hard-label cross-entropy
- Soft-target distillation
- Logit matching
- Feature distillation

### ROC plotting

**Utility:** `src/utils/plot_roc_curves.py`

This utility generates aggregated ROC curves and macro-AUC comparison plots for the teacher, baseline student, and KD student models.

Recommended usage from the repository root:

```bash
python -m src.utils.plot_roc_curves
```

Default output location:

```text
results/roc_curves/
```

### Experiment workflows

**Location:** `experiments/`

The grouped experiment suite is organised into dedicated experiment folders covering controlled studies such as KD vs baseline comparison, temperature sweeps, alpha sweeps, student-capacity sweeps, teacher-model comparison, distillation ablations, and robustness evaluation.

The experiment-level `run_all.sh` files are intended primarily for the Kelvin2 / SLURM workflow rather than as the standard local entry point. Repeated seeds and summary statistics within these experiments are handled by the experiment scripts themselves.

## Models

### Teacher

The teacher model is a multi-scale residual 1D CNN with:

- Squeeze-and-excitation attention
- Statistics pooling
- Higher channel capacity than the student
- Configurable capacity variants for teacher-comparison experiments

### Student

The student model is a lightweight residual 1D CNN with:

- Standard convolutional residual blocks
- Adaptive average pooling
- Smaller parameter count
- Lower latency and reduced model size
- Configurable small, medium, and large variants

## Dataset

This project uses the **PTB-XL** ECG dataset.

### Input format

- 12-lead ECG signals
- 500 Hz sampling rate
- 10-second recordings
- Processed into tensors of shape `(12, 5000)`

### Classification task

The task is 5-class diagnostic superclass classification:

- `CD`
- `HYP`
- `MI`
- `NORM`
- `STTC`

The raw PTB-XL dataset is not included in this repository due to size and distribution constraints. To reproduce the experiments, download PTB-XL separately and place it in the expected dataset location, typically:

```text
data/ptbxl/
```

## Experiments

The project is organised into a series of controlled experiments.

### Experiment 1 — KD vs Baseline

Compares:

- Teacher
- Baseline student
- KD student

### Experiment 2 — Temperature Sweep

Varies distillation temperature:

- `T = 1, 2, 4, 8, 16, 32`

### Experiment 3 — Alpha Sweep

Varies the weighting between:

- Hard-label supervision
- Teacher supervision

### Experiment 4 — Student Capacity Sweep

Compares:

- Small student
- Medium student
- Large student

### Experiment 5 — Teacher Model Comparison

Uses different teacher capacities to test how teacher strength affects student performance.

### Experiment 6 — Distillation Component Ablation

Compares:

- Cross-entropy only
- Soft targets only
- Feature distillation only
- Combined KD variants

### Experiment 7 — Robustness Evaluation

Evaluates model behaviour under perturbed inputs such as:

- Additive noise
- Amplitude scaling
- Missing leads
- Time masking

## Evaluation metrics

The software reports:

- Accuracy
- Macro-F1
- Weighted-F1
- Macro-AUC

**Macro-AUC** is the primary metric because it is more informative under class imbalance and better reflects class-balanced discrimination.

## Outputs

Depending on the script or experiment, outputs may include:

- Training logs
- Experiment logs
- Model checkpoints
- Printed metric summaries
- Experiment-level summary statistics
- ROC curves and generated figures

Common output locations:

- `logs/`
- `checkpoints/`
- `processed/`
- `results/`
- `experiments/.../logs/`
- `experiments/.../checkpoints/`

Generated logs, checkpoints, processed datasets, and large result artefacts are intentionally excluded from version control where appropriate, because they are reproducible outputs and may be too large to store in the repository.

## Typical workflow

### Local execution

A practical local workflow is:

1. Install dependencies
2. Download and place PTB-XL in `data/ptbxl/`
3. Run preprocessing
4. Train the teacher
5. Train the baseline student
6. Train the KD student
7. Generate evaluation outputs such as ROC curves, if required

This local route is mainly intended to verify that the environment, dataset paths, and main workflows operate correctly.

### Kelvin2 / SLURM execution

A practical Kelvin2 workflow is:

1. Set up the Python environment
2. Ensure PTB-XL is available in the expected location
3. Change into the repository root
4. Submit jobs using scripts in `slurm/`
5. Run grouped experiment workflows as required
6. Monitor logs in the relevant log directories

Example:

```bash
cd /users/40370279/csc4006/code
sbatch slurm/run_teacher.slurm
```

This is the preferred route for larger-scale experiment execution. The SLURM scripts are written to run from the project root so that logs, checkpoints, processed files, and result outputs are saved in the expected repository-level directories.

## Quick start

### Preprocess data

```bash
python scripts/preprocess_ptbxl.py
```

### Train teacher

```bash
python scripts/train_teacher.py
```

### Train baseline student

```bash
python scripts/train_student_baseline.py
```

### Train KD student

```bash
python scripts/train_student_kd.py
```

### Plot ROC curves

```bash
python -m src.utils.plot_roc_curves
```

### Submit cluster jobs

```bash
bash slurm/submit_all.sh
```

Or submit an individual job:

```bash
sbatch slurm/run_teacher.slurm
```

## Continuous integration

This repository includes a lightweight **GitHub Actions** CI workflow.

The CI workflow runs on an Ubuntu hosted runner using **Python 3.10** and performs three checks:

1. **Python syntax validation** using `compileall`
2. **Lightweight linting** using `ruff`
3. **Automated testing and coverage** using `pytest` and `pytest-cov`

The linting stage is intentionally focused on serious Python issues, such as syntax/parsing errors and undefined names, rather than enforcing a strict formatting style across the full research codebase.

Full PTB-XL preprocessing and model training are not executed in CI because they require the dataset and suitable GPU/HPC resources. Instead, the workflow validates importability, helper functions, dataset wrappers, augmentation safety, model forward passes, KD loss computation, metric utilities, CLI parsing, checkpoint metadata, and a synthetic KD training-step smoke test.

## Testing and quality assurance

The project includes several quality-assurance mechanisms:

- Version control using Git
- Python syntax validation through CI
- Lightweight Ruff linting through CI
- Automated unit tests through `pytest`
- Coverage reporting through `pytest-cov`
- Synthetic smoke testing of the core KD training path without PTB-XL or GPU access
- Shared evaluation utilities to reduce duplicated metric logic
- Fail-fast checks for important error conditions, such as missing processed data or missing teacher checkpoints
- Structured output locations for logs, checkpoints, and experiment results
- Documentation through `README.md`, `INSTALL.md`, `REPLICATION_GUIDE.md`, `TESTING.md`, and `docs/known_issues.md`

Run the lightweight QA suite locally with:

```bash
python -m compileall src scripts tests
ruff check src scripts tests --select E9,F63,F7,F82
pytest --cov=src --cov=scripts --cov-report=term-missing
```

The automated tests focus on smaller software components and helper functions rather than full ECG model training. This is appropriate for the project because complete training runs are computationally expensive and depend on external data and HPC resources. The full experimental workflows are validated through the documented Kelvin2/SLURM process, logs, checkpoints, and repeated-seed experiments.

For more detail, see `TESTING.md`.

## Reproducibility

The project supports reproducibility through:

- Fixed train/validation/test fold usage
- Explicit random seed control
- Checkpoint saving with metadata
- Experiment-specific logging
- SLURM-based batch execution
- Repeated-seed experiment workflows
- Automated CI checks for syntax, linting, unit tests, synthetic smoke tests, and coverage reporting
- Clear separation between source code and generated artefacts such as logs, checkpoints, processed data, and figures

Repeated seeds and summary statistics in the grouped experiment workflows are handled by the experiment scripts themselves rather than being treated as the main standalone user workflow.

## Important notes

- Some experiments depend on pretrained teacher checkpoints.
- Ensure dataset paths and checkpoint paths are correct before running jobs.
- Training, experiment, plotting, and SLURM workflows should ideally be launched from the repository root because many outputs are saved using project-root-relative paths such as `logs/`, `checkpoints/`, `processed/`, and `results/`.
- The main Python scripts support local execution of preprocessing and the core training workflows.
- Full grouped experiment workflows are organised under `experiments/` and are primarily intended for Kelvin2 / SLURM execution.
- Full model training is intentionally excluded from GitHub Actions CI because the workflow would require the PTB-XL dataset and significant compute resources.
- Generated outputs such as logs, checkpoints, processed datasets, and large result artefacts may be excluded from version control and regenerated using the documented workflows.
- For installation instructions, see `INSTALL.md`.
- For reproduction steps, see `REPLICATION_GUIDE.md`.
- For automated and manual QA details, see `TESTING.md`.
- For known issues and mitigations, see `docs/known_issues.md`.

## Supporting documentation

The repository includes:

- `README.md` — overview of the artefact, repository structure, workflows, CI, and outputs
- `INSTALL.md` — installation and environment setup instructions
- `REPLICATION_GUIDE.md` — steps for reproducing the main workflows and experiments
- `TESTING.md` — automated testing, CI, smoke testing, and manual validation guidance
- `docs/known_issues.md` — known limitations and mitigations for the software artefact
- `requirements.txt` — Python dependency list
- `LICENSE` — distribution rights for the artefact
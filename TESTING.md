# Testing and Quality Assurance

This repository includes lightweight automated quality-assurance checks for the research software artefact. The goal is to validate the reusable software components and core execution paths without requiring the full PTB-XL dataset or GPU/HPC resources in continuous integration.

## Automated checks

The automated QA workflow covers:

- Python syntax validation with `compileall`
- Lightweight Ruff linting for serious Python errors
- Unit tests with `pytest`
- Coverage reporting with `pytest-cov`
- Synthetic smoke tests for the model and KD training path

Run the full lightweight QA suite locally with:

```bash
python -m compileall src scripts tests
ruff check src scripts tests --select E9,F63,F7,F82
pytest --cov=src --cov=scripts --cov-report=term-missing
```

## Test coverage focus

The tests focus on components that can be checked quickly and deterministically:

- PTB-XL preprocessing helper functions
- Dataset wrapping and transform application
- ECG augmentation shape/dtype/finite-value safety
- Student and teacher model forward passes
- Knowledge-distillation loss computation and backpropagation
- A synthetic KD training-step smoke test
- Metric computation and empty-loader error handling
- Model parameter/size/checkpoint utilities
- CLI argument parsing for training scripts
- Checkpoint metadata used for reproducibility

## Synthetic smoke testing

Full training is not run during CI. Instead, the synthetic smoke test creates fake ECG tensors and validates that the teacher, student, projector, KD loss, optimiser step, and gradient flow operate together correctly. This confirms the core training path without depending on the PTB-XL dataset.

## Manual validation

Full experiment validation is performed outside CI because it requires the PTB-XL dataset and Kelvin2/GPU resources. Manual and HPC validation includes:

- verifying that preprocessing creates `processed/ptbxl_500hz_10s.npz`
- checking train/validation/test split summaries
- checking teacher, baseline student, and KD training logs
- confirming checkpoint creation and checkpoint metadata
- running repeated-seed experiment workflows
- comparing final metrics with the research article results

## What is not tested automatically

The following are intentionally excluded from GitHub Actions CI:

- downloading or storing PTB-XL
- full PTB-XL preprocessing
- full teacher/student/KD training runs
- SLURM job execution on Kelvin2
- GPU latency benchmarking
- large checkpoint generation

These exclusions are deliberate because the full experimental workflow depends on external data and high-performance computing resources. The CI workflow validates the codebase and lightweight executable paths; the replication guide documents how to reproduce the full experiments.

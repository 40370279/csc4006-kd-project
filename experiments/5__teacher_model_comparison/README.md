# Experiment 5: Teacher Model Comparison

## Goal
Evaluate how different teacher models affect KD student performance.

## Teachers
- Normal: teacher_cnn_best.pt
- Weak: teacher_cnn_weak_best.pt
- Strong: teacher_cnn_strong_best.pt

## Student setup
- student_size = medium
- alpha = 0.5
- temperature = 4.0
- seeds = 42, 123, 999

## Run
bash run_all.sh

## Output
- Logs: logs/
- Checkpoints: checkpoints/

## Notes
Only teacher model changes. Everything else is fixed.
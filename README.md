# To start Preprocessing
cd code
python3 -m src.preprocess_ptbxl

# To start training Teacher Model
cd code
python3 -m src.train_teacher
python3 -m src.train_student

# To start training Teacher/Student Model (Using SLURM on Kelvin2)
cd code
sbatch train_teacher_gpu.slurm
sbatch train_student_kd_gpu.slurm

# To check job queue
squeue -u $USER

# To see log output
tail -f logs/teacher_gpu_*.out


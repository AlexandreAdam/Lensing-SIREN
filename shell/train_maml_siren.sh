#!/bin/bash
#SBATCH --tasks=1
#SBATCH --cpus-per-task=3  # maximum cpu per task is 3.5 per gpus
#SBATCH --mem=32G			     # memory per node
#SBATCH --gres=gpu:1
#SBATCH --time=0-01:00		 # time (DD-HH:MM)
#SBATCH --account=rrg-lplevass
#SBATCH --job-name=Meta-Train-Siren
#SBATCH --output=%x-%j.out
source $LSIREN_PATH/venv/bin/activate
python $LSIREN_PATH/meta_train_siren.py\
  --dataset=hkappa188hst_TNG100_rau_spl.h5\
  --first_omega=30\
  --hidden_omega=30\
  --hidden_layers=1\
  --hidden_features=10\
  --learning_rate=1e-3\
  --step_size=1e-3\
  --loss_type=image\
  --learn_step_size\
  --per_param_step_size\
  --num_adaptation_steps=1\
  --batch_size=1\
  --max_batches=10\
  --use_cuda

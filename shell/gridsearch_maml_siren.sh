#!/bin/bash
#SBATCH --array=1-50
#SBATCH --tasks=1
#SBATCH --cpus-per-task=3  # maximum cpu per task is 3.5 per gpus
#SBATCH --mem=32G			     # memory per node
#SBATCH --gres=gpu:1
#SBATCH --time=0-10:00		 # time (DD-HH:MM)
#SBATCH --account=rrg-lplevass
#SBATCH --job-name=Meta-Train-Siren
#SBATCH --output=%x-%j.out
source $LSIREN_PATH/venv/bin/activate
python $LSIREN_PATH/gridsearch.py\
  --model_id=meta_siren_widesearch1\
  --n_models=50\
  --strategy=uniform\
  --epochs=50\
  --dataset=hkappa188hst_TNG100_rau_spl.h5\
  --first_omega 0.1 0.5 1 5 10 30 60\
  --hidden_omega 30\
  --hidden_layers 2 3 4 5\
  --hidden_features 64 128 256\
  --learning_rate 1e-3 1e-4 1e-5\
  --step_size 1e-3\
  --loss_type=image\
  --lr_type=global\
  --num_adaptation_steps 3 5 7 9\
  --batch_size 2 8 16\
  --epochs_til_checkpoint 2\
  --use_cuda\
  --max_time=10

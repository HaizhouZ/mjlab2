#!/bin/bash

#SBATCH --job-name=rl_training
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ilyass.taouil@tum.de
#SBATCH --mem=48gb
#SBATCH --time=4-00:00:00
#SBATCH --output=/rhome/itaouil/output/rl_training.log
#SBATCH --partition=submit
#SBATCH --gres=gpu:2
#SBATCH --constraint="a100"
#SBATCH --exclude=

#eval "$(conda shell.bash hook)"
#conda activate caption3d

echo Running on $(hostname)

#set -ex

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi

MUJOCO_GL=egl uv run torchrun \
  --nproc_per_node=2 \
  --no_python \
  train  Mjlab-Tracking-Flat-Unitree-G1-Box-No-State-Estimation\
    --distributed True \
    --motion-file motions/output/motion.npz \
  --env.scene.num_envs 8192 \
  --agent.max-iterations 30000
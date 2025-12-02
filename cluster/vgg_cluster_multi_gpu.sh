#!/bin/bash

#SBATCH --job-name=rl_training
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ilyass.taouil@tum.de
#SBATCH --mem=48gb
#SBATCH --time=4-00:00:00
#SBATCH --output=/rhome/itaouil/output/rl_training_multi_gpu.log
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

MUJOCO_GL=egl uv run train Mjlab-MultiTracking-Flat-Unitree-G1-Box-No-State-Estimation \
  --motion_dir motions/output/diverse  --env.scene.num_envs 8192 \
  --agent.max-iterations 30000 \
  --device cuda:0 \
  --agent.wandb-project single-vs-multi \
  --gpu-ids all

#!/bin/bash

#SBATCH --job-name=rl_training
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ilyass.taouil@tum.de
#SBATCH --mem=48gb
#SBATCH --time=4-00:00:00
#SBATCH --output=/rhome/itaouil/output/rl_training.log
#SBATCH --partition=submit
#SBATCH --gres=gpu:1
#SBATCH --constraint="rtx_a6000"
#SBATCH --exclude=

#eval "$(conda shell.bash hook)"
#conda activate caption3d

echo Running on $(hostname)

#set -ex

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi

MUJOCO_GL=egl uv run train Mjlab-Tracking-Flat-Unitree-G1-Box-No-State-Estimation \
  --motion-file motions/output/motion.npz \
  --env.scene.num_envs 4096 \
  --agent.max-iterations 30000 \
  --device cuda:0

#!/bin/bash
#SBATCH --open-mode=append
#SBATCH --output=/scratch/%u/learning-with-preferences/slurm_output/%x_%j.out
#SBATCH --error=/scratch/%u/learning-with-preferences/slurm_output/%x_%j.err
#SBATCH --export=ALL
#SBATCH --array=1-6
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=60GB
#SBATCH --mail-type=BEGIN,END,FAIL

# arg 1: command

singularity exec --nv --overlay /scratch/lhz209/conda_envs/pref/overlay-5GB-200K.ext3:ro /scratch/work/public/singularity/cuda11.7.99-cudnn8.5-devel-ubuntu22.04.2.sif bash -c """
source /ext3/env.sh
$(head -n $SLURM_ARRAY_TASK_ID scripts/generate_commands.txt | tail -n 1)

"""

# accelerate launch --num_processes 2 distributed_inference.py

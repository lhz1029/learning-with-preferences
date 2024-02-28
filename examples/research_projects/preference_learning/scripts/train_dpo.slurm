#!/bin/bash
#SBATCH --job-name=dpo_dry_run
#SBATCH --open-mode=append
#SBATCH --output=/scratch/%u/learning-with-preferences/slurm_output/%x_%j.out
#SBATCH --error=/scratch/%u/learning-with-preferences/slurm_output/%x_%j.err
#SBATCH --export=ALL
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=60GB
#SBATCH --mail-type=BEGIN,END,FAIL

singularity exec --nv --overlay /scratch/lhz209/conda_envs/pref/overlay-5GB-200K.ext3:ro /scratch/work/public/singularity/cuda11.7.99-cudnn8.5-devel-ubuntu22.04.2.sif bash -c """
source /ext3/env.sh

python /scratch/lhz209/learning-with-preferences/examples/research_projects/preference_learning/train_dpo.py \
        --model_name_or_path="checkpoints/sft_oasst_assistant/final_checkpoint" \
        --output_dir="checkpoints/dpo_dry_run" \
        --per_device_train_batch_size=5 \
        --gradient_accumulation_steps=2 \
        --per_device_eval_batch_size=10 \
        --eval_accumulation_steps=2 \
        --max_length=2048 \
        --logging_first_step=True \
        --eval_steps=1 \
        --max_steps=3 \
        --logging_steps=1 \
        --sanity_check=True
"""


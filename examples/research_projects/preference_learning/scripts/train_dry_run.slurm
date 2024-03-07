#!/bin/bash
#SBATCH --job-name=space
#SBATCH --open-mode=append
#SBATCH --output=/scratch/%u/learning-with-preferences/slurm_output/%x_%j.out
#SBATCH --error=/scratch/%u/learning-with-preferences/slurm_output/%x_%j.err
#SBATCH --export=ALL
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=60GB
#SBATCH --mail-type=BEGIN,END,FAIL

singularity exec --nv --overlay /scratch/lhz209/conda_envs/pref/overlay-5GB-200K.ext3:ro /scratch/work/public/singularity/cuda11.7.99-cudnn8.5-devel-ubuntu22.04.2.sif bash -c """
source /ext3/env.sh
python /scratch/lhz209/learning-with-preferences/examples/research_projects/preference_learning/train.py \
--model_name="meta-llama/Llama-2-7b-hf" \
--output_dir="checkpoints/dry_run" \
--roles="editor" \
--per_device_train_batch_size=1 \
--gradient_accumulation_steps=4 \
--evaluation_strategy="steps" \
--per_device_eval_batch_size=10 \
--eval_accumulation_steps=2 \
--packing=False \
--logging_steps=1 \
--max_seq_length=4096 \
--streaming=False \
--max_steps=6 \
--project_name="preference_roles" \
--logging_first_step=True \
--save_total_limit=2
"""
#!/bin/bash
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

# arg 1: roles

singularity exec --nv --overlay /scratch/lhz209/conda_envs/pref/overlay-5GB-200K.ext3:ro /scratch/work/public/singularity/cuda11.7.99-cudnn8.5-devel-ubuntu22.04.2.sif bash -c """
source /ext3/env.sh
python /scratch/lhz209/learning-with-preferences/examples/research_projects/preference_learning/train.py \
--model_name="meta-llama/Llama-2-7b-hf" \
--output_dir="checkpoints/sft_oasst_${1}_prompts" \
--roles=${1} \
--per_device_train_batch_size=5 \
--gradient_accumulation_steps=2 \
--evaluation_strategy="steps" \
--per_device_eval_batch_size=10 \
--eval_accumulation_steps=2 \
--packing=True \
--max_seq_length=2048 \
--logging_steps=20 \
--logging_first_step=True \
--streaming=False \
--project_name="preference_roles" \
--save_total_limit=5 \
--num_train_epochs=8 \
--metric_for_best_model=eval_valid_loss \
--greater_is_better=False
"""
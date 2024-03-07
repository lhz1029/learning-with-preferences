sbatch -J assistant /scratch/lhz209/learning-with-preferences/examples/research_projects/preference_learning/scripts/train_init.slurm assistant 4000
# sbatch -J judge /scratch/lhz209/learning-with-preferences/examples/research_projects/preference_learning/scripts/train_init.slurm judge 800
# sbatch -J editor /scratch/lhz209/learning-with-preferences/examples/research_projects/preference_learning/scripts/train_init.slurm editor 2000
# sbatch -J ae /scratch/lhz209/learning-with-preferences/examples/research_projects/preference_learning/scripts/train_init.slurm assistant,editor 3000
# sbatch -J aje /scratch/lhz209/learning-with-preferences/examples/research_projects/preference_learning/scripts/train_init.slurm assistant,judge,editor 1500
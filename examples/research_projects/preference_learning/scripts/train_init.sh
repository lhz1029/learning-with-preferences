sbatch -J assistant /scratch/lhz209/learning-with-preferences/examples/research_projects/preference_learning/scripts/train_init.slurm assistant
sbatch -J judge /scratch/lhz209/learning-with-preferences/examples/research_projects/preference_learning/scripts/train_init.slurm judge
sbatch -J editor /scratch/lhz209/learning-with-preferences/examples/research_projects/preference_learning/scripts/train_init.slurm editor
sbatch -J aje /scratch/lhz209/learning-with-preferences/examples/research_projects/preference_learning/scripts/train_init.slurm assistant,judge,editor
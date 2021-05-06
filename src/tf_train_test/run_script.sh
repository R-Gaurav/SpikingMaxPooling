#!/bin/bash
#SBATCH --time=00:20:00
#SBATCH --account=def-anarayan
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:p100:1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TEMP=$SLURM_TMPDIR
export TMP=$SLURM_TMPDIR
export TEMPDIR=$SLURM_TMPDIR
export TMPDIR=$SLURM_TMPDIR
python tf_train_test.py

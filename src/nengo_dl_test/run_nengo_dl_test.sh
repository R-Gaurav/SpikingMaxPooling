#!/bin/bash
#SBATCH --time=06:00:00
#SBATCH --account=def-anarayan
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:p100:1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TEMP=$SLURM_TMPDIR
export TMP=$SLURM_TMPDIR
export TEMPDIR=$SLURM_TMPDIR
export TMPDIR=$SLURM_TMPDIR
python nengo_dl_test.py --load_tf_wts=True

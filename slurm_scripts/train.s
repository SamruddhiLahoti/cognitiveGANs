#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --mem=16GB
#SBATCH --job-name=gan
#SBATCH --mail-type=END
#SBATCH --mail-user=samruddhi.lahoti@nyu.edu
#SBATCH --output=slurm_out/train.out

module purge

command="torchrun --nnodes=1 --nproc_per_node=1 train.py "

cd $SCRATCH/cognitiveGANs

singularity exec --nv \
	--overlay /$SCRATCH/pytorch/overlay-15GB-500K.ext3:ro \
	/scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif \
	/bin/bash -c "source /ext3/env.sh; $command;"

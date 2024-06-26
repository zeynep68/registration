#!/usr/bin/env bash
#SBATCH --account=jinm11
#SBATCH --time=02:00:00
#SBATCH --partition=dc-cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=128
#SBATCH --cpus-per-task=1
#SBATCH --output=tmp/log/out.%j
#SBATCH --error=tmp/log/err.%j

echo "Use working directory"
pwd

echo "Enabling environment..."
source /p/project/cjinm11/Private/oberstrass1/git/pli/pli-env/activate.sh

srun python cli/marching.py \
    /p/project/cjinm11/Private/hengelbrock2/git/master-segmentation/tmp/3D_volume/tissues_5_outer.h5 \
    tmp/vessels/vessels_5.ply


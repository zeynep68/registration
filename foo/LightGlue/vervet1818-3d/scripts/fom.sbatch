#!/usr/bin/env bash
#SBATCH --account=jinm11
#SBATCH --time=08:00:00
#SBATCH --partition=dc-cpu
#SBATCH --nodes=1
#SBATCH --output=logs/out.%j
#SBATCH --error=logs/err.%j

source ~/git/envs/pli-env/activate.sh


srun -n 128 pli_styles fom hsv \
-d 'data/aa/direction/*.h5' \
-i 'data/aa/inclination/*.h5' \
-m 'data/aa/masks/tissue/*.h5' \
-o data/aa/fom/ \
--saturation_min=127 \
--background_value=3 \
--direction_offset=21 \
--compression=gzip


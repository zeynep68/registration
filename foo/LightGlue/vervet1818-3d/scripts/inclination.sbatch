#!/usr/bin/env bash
#SBATCH --account=jinm11
#SBATCH --time=08:00:00
#SBATCH --partition=dc-cpu
#SBATCH --nodes=1
#SBATCH --output=logs/out.%j
#SBATCH --error=logs/err.%j

source ~/git/envs/pli-env/activate.sh


srun -n 128 pli_styles inclination transmittance_weighted \
-r 'data/aa/retardation/*.h5' \
-t 'data/aa/ntransmittance/*.h5' \
-o data/aa/inclination/ \
--method=plimage_smooth \
--t_M=0.21 \
--t_c=0.64 \
--r_ref_gm=0.12 \
--r_ref_wm=0.85 \
--median_kernel_size=9


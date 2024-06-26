#!/usr/bin/env bash
#SBATCH --account=jinm11
#SBATCH --time=01:00:00
#SBATCH --partition=dc-cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=128
#SBATCH --cpus-per-task=1
#SBATCH --output=tmp/log/out.%j
#SBATCH --error=tmp/log/err.%j

echo "Use working directory"
pwd

echo "Enabling environment..."
source meshlab-env/activate.sh


mkdir -p data/aa/mesh/pial/
mkdir -p data/aa/mesh/wm/

LEVEL=4

for LEVEL in 6 5 4; do
  srun bbrec mesh marching-cubes \
    data/aa/volume/pial/pial_${LEVEL}.h5 \
    data/aa/mesh/pial/pial_${LEVEL}.ply \
    --no-padding

  srun bbrec mesh marching-cubes \
    data/aa/volume/wm/wm_${LEVEL}_smooth.h5 \
    data/aa/mesh/wm/wm_${LEVEL}.ply \
    --no-padding
done

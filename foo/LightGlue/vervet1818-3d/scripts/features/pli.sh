#!/usr/bin/env bash
#SBATCH --account=jinm11
#SBATCH --time=06:00:00
#SBATCH --partition=dc-cpu
#SBATCH --nodes=1
#SBATCH --output=tmp/log/out.%j
#SBATCH --error=tmp/log/err.%j

echo "Use working directory"
pwd

echo "Enabling environment..."
source environment/activate.sh

srun -c 128 python cli/pli_glcm.py \
  --model_name=pli_glcm \
  --n_loaders=128


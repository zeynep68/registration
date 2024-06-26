#!/usr/bin/env bash
#SBATCH --account=jinm11
#SBATCH --time=04:00:00
#SBATCH --partition=dc-cpu
#SBATCH --nodes=1
#SBATCH --output=tmp/log/out.%j
#SBATCH --error=tmp/log/err.%j

echo "Use working directory"
pwd

echo "Enabling environment..."
source environment/activate.sh

srun -c 128 python cli/pli_pca.py \
  --model_name=pli_sobel_histo \
  --n_loaders=128 \
  --n_subsamples=64 \
  --n_samples=1000000 \
  --min_variance_explained=0.8 \
  --out_folder=data/aa/pca_80

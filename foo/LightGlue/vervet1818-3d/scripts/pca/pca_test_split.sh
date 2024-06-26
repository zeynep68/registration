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

for F in data/aa/features/simclr-*; do
  srun -c 128 python cli/perform_pca.py \
    --model_name=$(basename -- $F) \
    --data_group=Features/2048 \
    --n_subsamples=64 \
    --n_samples=1000000 \
    --min_variance_explained=0.8 \
    # --standardize
done


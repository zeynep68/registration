#!/usr/bin/env bash
#SBATCH --account=jinm11
#SBATCH --time=06:00:00
#SBATCH --partition=dc-cpu
#SBATCH --cpus-per-task=128
#SBATCH --nodes=1
#SBATCH --output=tmp/log/out.%j
#SBATCH --error=tmp/log/err.%j

echo "Use working directory"
pwd

echo "Enabling environment..."
source environment/activate.sh

srun python cli/perform_pca.py \
  --model_name=resnet50_planes8_circle_large \
  --data_group=Features/256 \
  --n_subsamples=64 \
  --n_samples=1000000 \
  --min_variance_explained=0.6

srun python cli/perform_pca.py \
  --model_name=resnet50_planes8_neighbor \
  --data_group=Features/256 \
  --n_subsamples=64 \
  --n_samples=1000000 \
  --min_variance_explained=0.6

srun python cli/perform_pca.py \
  --model_name=resnet50_planes8_sphere \
  --data_group=Features/256 \
  --n_subsamples=64 \
  --n_samples=1000000 \
  --min_variance_explained=0.6

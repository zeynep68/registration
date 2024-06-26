#!/usr/bin/env bash
#SBATCH --account=jinm11
#SBATCH --time=01:00:00
#SBATCH --partition=dc-cpu-devel
#SBATCH --cpus-per-task=128
#SBATCH --nodes=1
#SBATCH --output=tmp/log/out.%j
#SBATCH --error=tmp/log/err.%j

echo "Use working directory"
pwd

echo "Enabling environment..."
source environment/activate.sh


IN_FOLDER="data/aa/clusters/kmeans_pca_s2_128"

for F in $IN_FOLDER/*; do
  for NC in 2 5 9 21; do
    srun -n 1 python cli/agglomerative.py \
    --model_name=$(basename -- $F) \
    --n_clusters=$NC
  done 
done


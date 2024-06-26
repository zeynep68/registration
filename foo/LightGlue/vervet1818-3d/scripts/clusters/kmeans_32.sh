#!/usr/bin/env bash
#SBATCH --account=jinm11
#SBATCH --time=02:00:00
#SBATCH --partition=dc-cpu
#SBATCH --nodes=1
#SBATCH --output=tmp/log/out.%j
#SBATCH --error=tmp/log/err.%j

echo "Use working directory"
pwd

echo "Enabling environment..."
source environment/activate.sh


OUT_FOLDER="data/aa/clusters/kmeans_pca80_s1_32/"
SIGMA=1.
N_CLUSTER=32
N_SAMPLES=100000

# srun -c 128 -n 1 python cli/kmeans.py \
#   --model_name=pli_glcm_testset \
#   --out_folder=$OUT_FOLDER \
#   --feature_group=PCA \
#   --smooth_sigma=$SIGMA \
#   --n_clusters=$N_CLUSTER \
#   --n_samples=$N_SAMPLES

for F in data/aa/pca_80/simclr-imagenet*; do
  srun -c 128 -n 1 python cli/kmeans.py \
  --model_name=$(basename -- $F) \
  --out_folder=$OUT_FOLDER \
  --feature_group=PCA \
  --smooth_sigma=$SIGMA \
  --n_clusters=$N_CLUSTER \
  --n_samples=$N_SAMPLES
done

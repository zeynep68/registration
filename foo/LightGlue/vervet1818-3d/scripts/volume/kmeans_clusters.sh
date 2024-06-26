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


CLUSTERING=kmeans_pca80_s1_2
for F in data/aa/clusters/$CLUSTERING/simclr-imagene*; do
  FILENAME=$(basename -- $F)
  OUT=data/aa/volume/$CLUSTERING/$FILENAME
  mkdir -p $OUT
  srun -c 1 -n 1 pli volume stack $F $OUT/$CLUSTERING.h5 \
    --image_regex=".*s([0-9]{4}).*" \
    --fix_missing \
    --compression=gzip \
    --reindex='[1, 0, 2]' \
    --mirror='[false, true, true]'
  srun -n 1 pli convert nifti \
    $OUT/$CLUSTERING.h5 \
    -o $OUT
done

CLUSTERING=kmeans_pca80_s1_8
for F in data/aa/clusters/$CLUSTERING/simclr-imagene*; do
  FILENAME=$(basename -- $F)
  OUT=data/aa/volume/$CLUSTERING/$FILENAME
  mkdir -p $OUT
  srun -c 1 -n 1 pli volume stack $F $OUT/$CLUSTERING.h5 \
    --image_regex=".*s([0-9]{4}).*" \
    --fix_missing \
    --compression=gzip \
    --reindex='[1, 0, 2]' \
    --mirror='[false, true, true]'
  mkdir -p data/aa/volume/$FILENAME
  srun -n 1 pli convert nifti \
    $OUT/$CLUSTERING.h5 \
    -o $OUT
done


CLUSTERING=kmeans_pca80_s1_32
for F in data/aa/clusters/$CLUSTERING/simclr-imagene*; do
  FILENAME=$(basename -- $F)
  OUT=data/aa/volume/$CLUSTERING/$FILENAME
  mkdir -p $OUT
  srun -c 1 -n 1 pli volume stack $F $OUT/$CLUSTERING.h5 \
    --image_regex=".*s([0-9]{4}).*" \
    --fix_missing \
    --compression=gzip \
    --reindex='[1, 0, 2]' \
    --mirror='[false, true, true]'
  mkdir -p data/aa/volume/$FILENAME
  srun -n 1 pli convert nifti \
    $OUT/$CLUSTERING.h5 \
    -o $OUT
done


CLUSTERING=kmeans_pca80_s1_128
for F in data/aa/clusters/$CLUSTERING/simclr-imagene*; do
  FILENAME=$(basename -- $F)
  OUT=data/aa/volume/$CLUSTERING/$FILENAME
  mkdir -p $OUT
  srun -c 1 -n 1 pli volume stack $F $OUT/$CLUSTERING.h5 \
    --image_regex=".*s([0-9]{4}).*" \
    --fix_missing \
    --compression=gzip \
    --reindex='[1, 0, 2]' \
    --mirror='[false, true, true]'
  mkdir -p data/aa/volume/$FILENAME
  srun -n 1 pli convert nifti \
    $OUT/$CLUSTERING.h5 \
    -o $OUT
done


CLUSTERING=kmeans_pca80_s1_1024
for F in data/aa/clusters/$CLUSTERING/resnet50_planes8_962-1083_sphere_smal*; do
  FILENAME=$(basename -- $F)
  OUT=data/aa/volume/$CLUSTERING/$FILENAME
  mkdir -p $OUT
  srun -c 1 -n 1 pli volume stack $F $OUT/$CLUSTERING.h5 \
    --image_regex=".*s([0-9]{4}).*" \
    --fix_missing \
    --compression=gzip \
    --reindex='[1, 0, 2]' \
    --mirror='[false, true, true]'
  mkdir -p data/aa/volume/$FILENAME
  srun -n 1 pli convert nifti \
    $OUT/$CLUSTERING.h5 \
    -o $OUT
done

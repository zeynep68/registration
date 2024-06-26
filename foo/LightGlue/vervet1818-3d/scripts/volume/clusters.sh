#!/usr/bin/env bash
#SBATCH --account=jinm11
#SBATCH --time=02:00:00
#SBATCH --partition=dc-cpu
#SBATCH --cpus-per-task=128
#SBATCH --nodes=1
#SBATCH --output=tmp/log/out.%j
#SBATCH --error=tmp/log/err.%j

echo "Use working directory"
pwd

echo "Enabling environment..."
source environment/activate.sh


for F in data/aa/clusters/agglomerative_2/*; do
  OUT=data/aa/volume/agglomerative_2/$(basename -- $F)
  mkdir -p $OUT
  srun -n 1 pli volume stack $F $OUT/agglomerative_2.h5 \
    --image_regex=".*s([0-9]{4}).*" \
    --fix_missing \
    --compression=gzip \
    --reindex='[1, 0, 2]' \
    --mirror='[false, true, true]'
done


for F in data/aa/clusters/agglomerative_5/*; do
  OUT=data/aa/volume/agglomerative_5/$(basename -- $F)
  mkdir -p $OUT
  srun -n 1 pli volume stack $F $OUT/agglomerative_5.h5 \
    --image_regex=".*s([0-9]{4}).*" \
    --fix_missing \
    --compression=gzip \
    --reindex='[1, 0, 2]' \
    --mirror='[false, true, true]'
done


for F in data/aa/clusters/agglomerative_9/*; do
  OUT=data/aa/volume/agglomerative_9/$(basename -- $F)
  mkdir -p $OUT
  srun -n 1 pli volume stack $F $OUT/agglomerative_9.h5 \
    --image_regex=".*s([0-9]{4}).*" \
    --fix_missing \
    --compression=gzip \
    --reindex='[1, 0, 2]' \
    --mirror='[false, true, true]'
done

for F in data/aa/clusters/agglomerative_21/*; do
  OUT=data/aa/volume/agglomerative_21/$(basename -- $F)
  mkdir -p $OUT
  srun -n 1 pli volume stack $F $OUT/agglomerative_21.h5 \
    --image_regex=".*s([0-9]{4}).*" \
    --fix_missing \
    --compression=gzip \
    --reindex='[1, 0, 2]' \
    --mirror='[false, true, true]'
done


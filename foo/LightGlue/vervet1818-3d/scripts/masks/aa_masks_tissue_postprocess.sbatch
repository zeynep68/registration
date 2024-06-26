#!/usr/bin/env bash
#SBATCH --account=jinm11
#SBATCH --time=02:00:00
#SBATCH --partition=dc-cpu
#SBATCH --nodes=1
#SBATCH --output=tmp/log/out.%j
#SBATCH --error=tmp/log/err.%j

echo "Use working directory"
pwd

export CUDA_VISIBLE_DEVICES=0,1,2,3
# export NCCL_DEBUG=INFO

echo "Enabling environment..."
source tools/vervet-tissue-segmentation/env/activate.sh

srun -n 32 python tools/vervet-tissue-segmentation/cli/postprocess.py \
    data/aa/masks/tissue/* \
    --wm_count=300000 \
    --gm_count=1000000 \
    --bg_count=10000 \
    --wm_fill_count=100000 \
    --chunk_size=256 \
    --compression=gzip \


#!/usr/bin/env bash
#SBATCH --account=jinm11
#SBATCH --time=08:00:00
#SBATCH --partition=dc-gpu
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --output=tmp/log/out.%j
#SBATCH --error=tmp/log/err.%j

echo "Use working directory"
pwd

export CUDA_VISIBLE_DEVICES=0,1,2,3
# export NCCL_DEBUG=INFO

mkdir -p data/aa/masks/tissue/

echo "Enabling environment..."
source tools/vervet-tissue-segmentation/env/activate.sh

srun -n 8 python tools/vervet-tissue-segmentation/cli/apply-unet.py \
    --ckpt=tools/vervet-tissue-segmentation/data/ckpt/pliunet-v6/version_3/pliunet-v6_3_epoch=99_train_loss=0.00175.ckpt \
    --trans=in/volume-reconstruction/Transmittance_Retardation/nifti/*NTransmittance* \
    --dir=in/volume-reconstruction/Transmittance_Retardation/nifti/*Direction* \
    --ret=in/volume-reconstruction/Transmittance_Retardation/nifti/*Retardation* \
    --out=data/aa/masks/tissue \
    --num_workers=16 \
    --batch_size=16 \
    --patch_size=796 \
    --out_size=420 \
    --ram \
    
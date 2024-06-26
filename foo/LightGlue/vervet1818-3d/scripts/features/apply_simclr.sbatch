#!/usr/bin/env bash
#SBATCH --account=jinm11
#SBATCH --time=04:00:00
#SBATCH --partition=dc-gpu
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --output=tmp/log/out.%j
#SBATCH --error=tmp/log/err.%j


CKPT="models/simclr_converter/resnet50-1x.pth"
MODEL="simclr-imagenet"

echo "Use working directory"
pwd

export CUDA_VISIBLE_DEVICES=0,1,2,3
# export NCCL_DEBUG=INFO

mkdir -p data/aa/features/$MODEL

echo "Enabling environment..."
source tools/vervet-contrastive-3d/environment/activate.sh


srun -c 16 -n 8 python scripts/features/apply_simclr.py \
  --ckpt=$CKPT \
  --trans=in/volume-reconstruction/Transmittance_Retardation/nifti/Vervet1818aa_*NTransmittance.nii.gz \
  --dir=in/volume-reconstruction/Transmittance_Retardation/nifti/Vervet1818aa_*Direction.nii.gz \
  --ret=in/volume-reconstruction/Transmittance_Retardation/nifti/Vervet1818aa_*Retardation.nii.gz \
  --out=data/aa/features/${MODEL}/ \
  --num_workers=16 \
  --batch_size=64 \
  --overlap=0.5 \
  --ram \
  --dtype=float16

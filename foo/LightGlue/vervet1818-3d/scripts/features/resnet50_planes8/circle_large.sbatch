#!/usr/bin/env bash
#SBATCH --account=jinm11
#SBATCH --time=04:00:00
#SBATCH --partition=dc-gpu
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --output=tmp/log/out.%j
#SBATCH --error=tmp/log/err.%j

MODEL=resnet50_planes8_circle_large
VERSION=version_1
CKPT=last.ckpt

echo "Use working directory"
pwd

export CUDA_VISIBLE_DEVICES=0,1,2,3
# export NCCL_DEBUG=INFO

mkdir -p data/aa/features/$MODEL

echo "Enabling environment..."
source tools/vervet-contrastive-3d/environment/activate.sh


srun -n 8 python tools/vervet-contrastive-3d/scripts/apply-encoder.py \
  --ckpt=tools/vervet-contrastive-3d/models/$MODEL/$VERSION/checkpoints/$CKPT \
  --trans=in/volume-reconstruction/Transmittance_Retardation/nifti/Vervet1818aa_*NTransmittance.nii.gz \
  --dir=in/volume-reconstruction/Transmittance_Retardation/nifti/Vervet1818aa_*Direction.nii.gz \
  --ret=in/volume-reconstruction/Transmittance_Retardation/nifti/Vervet1818aa_*Retardation.nii.gz \
  --out=data/aa/features/$MODEL/ \
  --num_workers=16 \
  --batch_size=16 \
  --overlap=0.5 \
  --ram \
  --dtype=float16

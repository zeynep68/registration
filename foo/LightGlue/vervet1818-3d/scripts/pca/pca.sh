#!/usr/bin/env bash
#SBATCH --account=jinm11
#SBATCH --time=12:00:00
#SBATCH --partition=dc-cpu
#SBATCH --cpus-per-task=128
#SBATCH --nodes=1
#SBATCH --output=tmp/log/out.%j
#SBATCH --error=tmp/log/err.%j

echo "Use working directory"
pwd

echo "Enabling environment..."
source environment/activate.sh

#srun python cli/perform_pca.py \
#  --model_name=resnet18_circle \
#  --data_group=Features/512 \

#srun python cli/perform_pca.py \
#  --model_name=resnet18_circle_large \
#  --data_group=Features/512 \

#srun python cli/perform_pca.py \
#  --model_name=resnet18_neighbor \
#  --data_group=Features/512 \

#srun python cli/perform_pca.py \
#  --model_name=resnet18_same \
#  --data_group=Features/512 \

#srun python cli/perform_pca.py \
#  --model_name=resnet18_sphere \
#  --data_group=Features/512 \

#srun python cli/perform_pca.py \
#  --model_name=resnet18_sphere_large \
#  --data_group=Features/512 \

#srun python cli/perform_pca.py \
#  --model_name=resnet18_sphere_small \
#  --data_group=Features/512 \

#srun python cli/perform_pca.py \
#  --model_name=resnet18_touching \
#  --data_group=Features/512 \

#srun python cli/perform_pca.py \
#  --model_name=resnet34_circle \
#  --data_group=Features/512 \

srun python cli/perform_pca.py \
  --model_name=resnet34_circle_large \
  --data_group=Features/512 \

srun python cli/perform_pca.py \
  --model_name=resnet34_neighbor \
  --data_group=Features/512 \

#srun python cli/perform_pca.py \
#  --model_name=resnet34_same \
#  --data_group=Features/512 \

#srun python cli/perform_pca.py \
#  --model_name=resnet34_sphere \
#  --data_group=Features/512 \

#srun python cli/perform_pca.py \
#  --model_name=resnet34_sphere_large \
#  --data_group=Features/512 \

srun python cli/perform_pca.py \
  --model_name=resnet34_sphere_small \
  --data_group=Features/512 \

#srun python cli/perform_pca.py \
#  --model_name=resnet34_touching \
#  --data_group=Features/512 \

#srun python cli/perform_pca.py \
#  --model_name=resnet50_planes8_circle \
#  --data_group=Features/256 \

srun python cli/perform_pca.py \
  --model_name=resnet50_planes8_circle_large \
  --data_group=Features/256 \

#srun python cli/perform_pca.py \
#  --model_name=resnet50_planes8_neighbor \
#  --data_group=Features/256 \

#srun python cli/perform_pca.py \
#  --model_name=resnet50_planes8_overlap \
#  --data_group=Features/256 \

#srun python cli/perform_pca.py \
#  --model_name=resnet50_planes8_same \
#  --data_group=Features/256 \

#srun python cli/perform_pca.py \
#  --model_name=resnet50_planes8_sphere \
#  --data_group=Features/256 \

#srun python cli/perform_pca.py \
#  --model_name=resnet50_planes8_sphere_large \
#  --data_group=Features/256 \

#srun python cli/perform_pca.py \
#  --model_name=resnet50_planes8_sphere_small \
#  --data_group=Features/256 \

#srun python cli/perform_pca.py \
#  --model_name=resnet50_planes8_touching \
#  --data_group=Features/256 \

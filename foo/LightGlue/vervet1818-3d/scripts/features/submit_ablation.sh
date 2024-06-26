#!/usr/bin/env bash

MODEL=(
    "cl-2d_all_augmentations_circle_small"
    "cl-2d_no_augmentations_circle_small"
    "cl-2d_only_affine_circle_small"
    "cl-2d_only_blur_circle_small"
    "cl-2d_only_flip_circle_small"
    "cl-2d_only_scale-attenuation_circle_small"
    "cl-2d_only_scale-thickness_circle_small"
    "cl-3d_all_augmentations_sphere_small"
    "cl-3d_no_augmentations_sphere_small"
    "cl-3d_only_affine_sphere_small"
    "cl-3d_only_blur_sphere_small"
    "cl-3d_only_flip_sphere_small"
    "cl-3d_only_scale-attenuation_sphere_small"
    "cl-3d_only_scale-thickness_sphere_small"
)

VERSION=version_1
CKPT=last.ckpt

for m in ${MODEL[*]}; do
  echo Submit $m $VERSION $CKPT
  sbatch scripts/features/apply_encoder.sbatch $m $VERSION $CKPT
done


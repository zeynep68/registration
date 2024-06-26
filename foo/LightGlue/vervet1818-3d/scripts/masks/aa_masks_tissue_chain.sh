#!/bin/bash

ID=$(sbatch --parsable scripts/masks/aa_masks_tissue.sbatch)
sbatch --dependency=afterany:${ID} scripts/masks/aa_masks_tissue_postprocess.sbatch

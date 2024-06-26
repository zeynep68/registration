source environment/activate.sh
mkdir -p data/aa/volume/fom/
pli volume stack data/aa/david_fom/ data/aa/volume/fom/david_fom_6_fixed.h5 \
  --image_pyramid=6 \
  --mask_path=data/aa/masks/cortex/ \
  --mask_pyramid=6 \
  --mask_reindex='[1, 0]' \
  --mask_mirror='[false, true]' \
  --background_mask=3 \
  --fix_missing \
  --compression=lzf \
  --reindex='[2, 0, 1]' \
  --mirror='[true, true, true]'

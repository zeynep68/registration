source environment/activate.sh
mkdir -p data/aa/volume/fom/
pli volume stack data/aa/fom/ data/aa/volume/fom/fom_5.h5 \
  --image_pyramid=5 \
  --mask_path=data/aa/masks/cortex/ \
  --mask_pyramid=5 \
  --background_mask=3 \
  --fix_missing \
  --compression=gzip \
  --reindex='[1, 0, 2]' \
  --mirror='[true, true, true]'

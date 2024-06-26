source environment/activate.sh
mkdir -p data/aa/volume/cortex/
pli volume stack data/aa/masks/cortex/ data/aa/volume/cortex/cortex_6_fixed.h5 \
  --image_pyramid=6 \
  --fix_missing \
  --compression=gzip \
  --reindex='[1, 0, 2]' \
  --mirror='[false, true, true]'

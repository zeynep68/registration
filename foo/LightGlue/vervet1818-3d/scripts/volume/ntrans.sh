source environment/activate.sh
mkdir -p data/aa/volume/ntrans/
pli volume stack data/aa/ntransmittance/ data/aa/volume/ntrans/ntrans_5.h5 \
  --image_pyramid=5 \
  --mask_path=data/aa/masks/cortex/ \
  --mask_pyramid=5 \
  --background_mask=3 \
  --fix_missing \
  --compression=gzip \
  --reindex='[1, 0, 2]' \
  --mirror='[true, true, true]'

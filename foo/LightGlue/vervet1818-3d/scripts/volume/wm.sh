source environment/activate.sh
mkdir -p data/aa/volume/wm/
pli volume clip data/aa/volume/cortex/cortex_6.h5 data/aa/volume/wm/wm_6.h5 --true_values [1]
pli volume clip data/aa/volume/cortex/cortex_5.h5 data/aa/volume/wm/wm_5.h5 --true_values [1]
pli volume clip data/aa/volume/cortex/cortex_4.h5 data/aa/volume/wm/wm_4.h5 --true_values [1]

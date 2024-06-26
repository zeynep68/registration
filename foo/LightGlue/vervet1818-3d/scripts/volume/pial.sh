source environment/activate.sh
mkdir -p data/aa/volume/pial/
pli volume clip data/aa/volume/cortex/cortex_6.h5 data/aa/volume/pial/pial_6.h5 --true_values "[1, 2]"
pli volume clip data/aa/volume/cortex/cortex_5.h5 data/aa/volume/pial/pial_5.h5 --true_values "[1, 2]"
pli volume clip data/aa/volume/cortex/cortex_4.h5 data/aa/volume/pial/pial_4.h5 --true_values "[1, 2]"

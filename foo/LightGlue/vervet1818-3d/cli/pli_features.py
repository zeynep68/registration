import os
import re

import pandas as pd
import numpy as np
import h5py as h5
from tqdm import tqdm

import pli

import click


# Paths
trans_path = "data/aa/ntransmittance/"
dir_path = "data/aa/direction/"
ret_path = "data/aa/retardation/"

# Group of the mask in the H5 files
ft_group = 'Features'

# Corresponding pyramid for features
feature_pyramid = 6
pli_pyramid = 1

seed = 299792458

# Number of bins for pli features
n_bins = 42


@click.command()
@click.option("--model_name", default='pli_sobel_histo', type=str)
@click.option("--n_loaders", default=1, type=int, help="Number of loaders")
@click.option("--out_folder", default="data/aa/features/", help="Folder to store PCA results")
def create_pca(model_name, n_loaders, out_folder):

    p = re.compile('.*s([0-9]{4})_.*')

    feature_list = []
    for tf, df, rf in zip(sorted(os.listdir(trans_path)), sorted(os.listdir(dir_path)), sorted(os.listdir(ret_path))):
        id = int(p.match(tf)[1])
        id_2 = int(p.match(rf)[1])
        assert id == id_2

        trans_section = pli.data.Section(os.path.join(trans_path, tf))
        spacing = tuple((2 ** feature_pyramid) * s for s in trans_section.spacing)
        origin = trans_section.origin
        trans_section.close_file_handle()

        feature_list.append({'id': id, 'spacing': spacing, 'origin': origin,
                             'file_trans': os.path.join(trans_path, tf),
                             'file_dir': os.path.join(dir_path, df),
                             'file_ret': os.path.join(ret_path, rf)})
    files_df = pd.DataFrame(feature_list).sort_values('id').reset_index(drop=True)

    # Fit PCA by subsample of the data
    from vervet1818_3d.utils.io import pli2histo

    out_path = os.path.join(out_folder, model_name)

    if not os.path.exists(out_path):
        os.makedirs(out_path, exist_ok=True)

    for k, r in tqdm(files_df.iterrows(), total=len(files_df)):
        trans_section = pli.data.Section(r.file_trans)
        dir_section = pli.data.Section(r.file_dir)
        ret_section = pli.data.Section(r.file_ret)

        test_features = pli2histo(
            trans=trans_section.pyramid[pli_pyramid][:],
            dir=np.deg2rad(dir_section.pyramid[pli_pyramid]),
            ret=ret_section.pyramid[pli_pyramid][:],
            pyramid=feature_pyramid - pli_pyramid,
            n_bins=n_bins,
            n_loaders=n_loaders,
        ).transpose(2, 0, 1)

        out_file = os.path.basename(r.file_ret).replace('Retardation.nii', 'PLI')
        with h5.File(os.path.join(out_path, out_file), 'w') as f:

            ft_key = f"{ft_group}/{test_features.shape[0]}"

            pca_dset = f.create_dataset(ft_key, data=test_features, dtype=np.float32)
            pca_dset.attrs['spacing'] = r.spacing
            pca_dset.attrs['origin'] = r.origin


if __name__ == "__main__":
    create_pca()

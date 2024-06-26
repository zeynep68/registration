import os
import re

import pandas as pd
import numpy as np
import h5py as h5

from tqdm import tqdm
import matplotlib.pyplot as plt

import pli
import pli.image as im

import click


# Paths
trans_path = "data/aa/ntransmittance/"
dir_path = "data/aa/direction/"
ret_path = "data/aa/retardation/"
mask_path = "data/aa/masks/cortex/"

# Group of the mask in the H5 files
mask_group = 'Image'
pca_key = "PCA"
valid_key = "Valid"

# Masking of features to include foreground only
mask_features = True
background_class = 3

# Coressponding pyramid
mask_pyramid = 6
pli_pyramid = 1

seed = 299792458

# Number of bins for pli features
n_bins = 42

# Sections to test on (From Takemura paper)
test_ids = [860, 898, 961, 1061]


@click.command()
@click.option("--model_name", default='pli_histo', type=str)
@click.option("--n_subsamples", default=64, type=int, help="Number of randomly selected sections")
@click.option("--n_loaders", default=1, type=int, help="Number of randomly selected sections")
@click.option("--n_samples", default=1_000_000, type=int, help="Number of samples used to fit PCA")
@click.option("--min_variance_explained", default=0.6, help="Minimum variance explained by selected components")
@click.option("--out_folder", default="data/aa/pca/", help="Folder to store PCA results")
def create_pca(model_name, n_subsamples, n_loaders, n_samples, min_variance_explained, out_folder):

    p = re.compile('.*s([0-9]{4})_.*')
    
    feature_list = []
    for tf, df, rf in zip(sorted(os.listdir(trans_path)), sorted(os.listdir(dir_path)), sorted(os.listdir(ret_path))):
        id = int(p.match(tf)[1])
        id_2 = int(p.match(rf)[1])
        assert id == id_2
        feature_list.append({'id': id, 'file_trans': os.path.join(trans_path, tf),
                             'file_dir': os.path.join(dir_path, df),
                             'file_ret': os.path.join(ret_path, rf)})
    feature_df = pd.DataFrame(feature_list).sort_values('id').reset_index(drop=True)

    mask_list = []
    for f in sorted(os.listdir(mask_path)):
        id = int(p.match(f)[1])
        mask_section = pli.data.Section(os.path.join(mask_path, f))
        spacing = tuple((2 ** mask_pyramid) * s for s in mask_section.spacing)
        origin = mask_section.origin
        mask_section.close_file_handle()
        mask_list.append({'id': id, 'spacing': spacing, 'origin': origin, 'file_mask': os.path.join(mask_path, f)})
    mask_df = pd.DataFrame(mask_list).sort_values('id').reset_index(drop=True)

    files_df = mask_df.merge(feature_df, on='id', how='left')

    # Fit PCA by subsample of the data
    from vervet1818_3d.utils.io import pli2feat

    np.random.seed(seed)

    selected_features = []
    selected_masks = []

    section_ids = []

    for k, r in tqdm(files_df.sample(n_subsamples).sort_values('id').iterrows(), total=n_subsamples):
        trans_section = pli.data.Section(r.file_trans)
        dir_section = pli.data.Section(r.file_dir)
        ret_section = pli.data.Section(r.file_ret)

        features = pli2feat(
            trans=trans_section.pyramid[pli_pyramid][:],
            dir=np.deg2rad(dir_section.pyramid[pli_pyramid]),
            ret=ret_section.pyramid[pli_pyramid][:],
            pyramid=mask_pyramid - pli_pyramid,
            n_bins=n_bins,
            n_loaders=n_loaders,
        )

        mask_section = pli.data.Section(r.file_mask, data_group=mask_group)
        mask = mask_section.pyramid[mask_pyramid][:]

        selected_features.append(features)
        selected_masks.append(mask)
        section_ids.append(r.id)

    print(f"Use sections {section_ids}")

    if mask_features:
        valid_features = [f[m != background_class] for f, m in zip(selected_features, selected_masks)]
    else:
        valid_features = [sf.reshape(-1, sf.shape[-1]) for sf in selected_features]

    valid_lengths = [len(vf) for vf in valid_features]
    valid_features = np.vstack(valid_features)
    del selected_features

    # Perform PCA
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    np.random.seed(seed)

    ix = np.random.choice(np.arange(len(valid_features)), n_samples)
    pca_components = len(valid_features[0])

    feature_selection = valid_features[ix].astype(np.float64)

    # PLI features might be scaled differently, we standardize them
    scaler = StandardScaler()
    std_features = scaler.fit_transform(feature_selection)

    pca = PCA(n_components=pca_components, whiten=False)
    pca.fit(std_features)

    # Get optimal number of components
    from vervet1818_3d.utils.stats import profile_likelihood

    pl = np.array([profile_likelihood(l, pca) for l in range(pca.n_components)])

    pl_components = np.argmax(pl)
    print(f"Optimum found at {pl_components} components")

    plt.vlines(np.argmax(pl), pl.max(), pl.min(), colors='black', linestyles='--')
    plt.plot(pl[:100])
    plt.show()

    # Function of variance explained
    scree = np.cumsum(pca.explained_variance_)
    scree /= scree[-1]

    variance_components = np.argwhere(scree > min_variance_explained)[0, 0] + 1

    n_components = max(pl_components, variance_components)

    variance_explained = scree[n_components -1] / scree[-1]

    print(f"Explained variance: {100 * variance_explained:.2f}% with {n_components} components")
    plt.plot(scree[:100])
    plt.vlines(n_components, scree.max(), 0, colors='black', linestyles='--')
    plt.title("Proportion of variance explained")
    plt.xlabel("#Components")
    plt.grid()
    plt.show()

    # Store PCAs
    from vervet1818_3d.utils.stats import pca_n_transform

    out_path = os.path.join(out_folder, model_name)

    if not os.path.exists(out_path):
        os.mkdir(out_path)

    # Save PCA related components
    with h5.File(os.path.join(out_path, "components.h5"), 'w') as f:
        pca_components = f.create_dataset("components", data=pca.components_, dtype=np.float32)
        pca_components.attrs['explained_variance'] = pca.explained_variance_
        pca_components.attrs['scaler_mean'] = scaler.mean_
        pca_components.attrs['scaler_var'] = scaler.var_
        pca_components.attrs['scaler_scale'] = scaler.scale_
        pca_components.attrs['n_bins'] = n_bins

    for k, r in tqdm(files_df.iterrows(), total=len(files_df)):
        trans_section = pli.data.Section(r.file_trans)
        dir_section = pli.data.Section(r.file_dir)
        ret_section = pli.data.Section(r.file_ret)

        test_features = pli2feat(
            trans=trans_section.pyramid[pli_pyramid][:],
            dir=np.deg2rad(dir_section.pyramid[pli_pyramid]),
            ret=ret_section.pyramid[pli_pyramid][:],
            pyramid=mask_pyramid - pli_pyramid,
            n_bins=n_bins,
            n_loaders=n_loaders,
        )

        mask_section = pli.data.Section(r.file_mask, data_group=mask_group)
        test_mask = mask_section.pyramid[mask_pyramid][:]

        test_valid = test_mask != background_class
        test_std = scaler.transform(test_features.reshape(-1, test_features.shape[-1]))

        test_pca = pca_n_transform(test_std, pca, n_components)
        test_pca = test_pca.reshape(*test_features.shape[:2], test_pca.shape[-1])

        out_file = os.path.basename(r.file_ret).replace('Retardation.nii', 'PCA')
        with h5.File(os.path.join(out_path, out_file), 'w') as f:
            pca_dset = f.create_dataset(pca_key, data=test_pca.transpose(2, 0, 1), dtype=np.float32)
            pca_dset.attrs['spacing'] = r.spacing
            pca_dset.attrs['origin'] = r.origin
            pca_dset.attrs['variance_explained'] = variance_explained
            pca_dset.attrs['eigenvalues'] = pca.explained_variance_
            pca_dset.attrs['n_components'] = n_components
            pca_dset.attrs['pl_components'] = pl_components

            valid_dset = f.create_dataset(valid_key, data=test_valid)
            valid_dset.attrs['spacing'] = r.spacing
            valid_dset.attrs['origin'] = r.origin

    # Print thumbnails for selected sections
    out_path_thmb = os.path.join(out_folder, model_name, 'overview')

    for test_id in test_ids:
        test_file = files_df.loc[files_df.id == test_id].iloc[0]

        mask_section = pli.data.Section(test_file.file_mask, data_group=mask_group)
        test_mask = mask_section.pyramid[mask_pyramid][:]

        pca_section = pli.data.Section(test_file.file_pca, data_group=pca_key)
        test_pca = pca_section.image[:]
        n_components = pca_section.attrs['n_components']

        test_pca = np.rot90(test_pca, 1, axes=(-1, -2))
        test_mask = np.rot90(test_mask, 1, axes=(-1, -2))

        test_tissue = test_mask != background_class
        test_valid = test_tissue

        print_pca = np.empty((n_components, *test_mask.shape[-2:]), dtype=np.float16)
        for i in range(len(print_pca)):
            print_pca[i] = test_pca[i]

        vmin = np.min(print_pca[:, ~test_valid])
        vmax = np.max(print_pca[:, ~test_valid])
        print_pca[:, ~test_valid] = 0

        for i in range(len(print_pca)):
            plt.imsave(out_path_thmb + f"/pca_{n_components}_s{test_id:04d}_{i}.png", print_pca[i], cmap='magma', vmax=vmax, vmin=vmin)


if __name__ == "__main__":
    create_pca()

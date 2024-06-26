import os

import re
import pandas as pd
import numpy as np

import h5py as h5

import pli
import pli.image as im

import click

from tqdm import tqdm


# Paths
feature_path = f"data/aa/pca_80/"
mask_path = "data/aa/masks/cortex/"

# Masking of features to include foreground only
mask_features = True
background_class = 3

# Groups
mask_group = 'Image'
out_modality = 'Mask'

# Coressponding pyramid to the Feature size
mask_pyramid = 6

# Out parameters
chunk_size = 128
compression ='gzip'

seed = 299792458 


@click.command()
@click.option("--model_name", default='pli_histo', type=str)
@click.option("--out_folder", default='data/aa/clusters/kmeans', type=str)
@click.option("--feature_group", default='PCA', type=str)
@click.option("--smooth_sigma", default=1.0, type=float, help="Sigma of Gaussian smoothing kernel applied to fmaps")
@click.option("--n_clusters", default=256, type=int, help="Number of clusters for KMeans")
@click.option("--n_samples", default=64_000, type=int, help="Number of samples used to fit KMeans")
def create_clusters(model_name, out_folder, feature_group, smooth_sigma, n_clusters, n_samples):

    # Load section infos
    feature_folder = os.path.join(feature_path, model_name)
    
    p = re.compile('.*s([0-9]{4})_.*h5')

    feature_list = []
    for f in sorted(os.listdir(feature_folder)):
        match = p.match(f)
        if match:
            id = int(match[1])
            with h5.File(os.path.join(feature_folder, f)) as h5f:
                spacing = h5f[feature_group].attrs['spacing']
                origin = h5f[feature_group].attrs['origin']
            feature_list.append({'id': id, 'spacing': spacing, 'origin': origin,
                                 'file_features': os.path.join(feature_folder, f)})
    feature_df = pd.DataFrame(feature_list)

    mask_list = []
    for f in sorted(os.listdir(mask_path)):
        match = p.match(f)
        if match:
            id = int(match[1])
            mask_list.append({'id': id, 'file_mask': os.path.join(mask_path, f)})
    mask_df = pd.DataFrame(mask_list)

    files_df = mask_df.merge(feature_df, on='id', how='inner').sort_values('id').reset_index(drop=True)
    print(files_df.head())

    # Load feature maps
    from skimage import filters
    from vervet1818_3d.utils.io import read_masked_features

    selected_features = []
    selected_masks = []

    for k, r in tqdm(files_df.sort_values('id').iterrows(), total=len(files_df)):
        features, mask = read_masked_features(
            r.file_features,
            r.file_mask,
            mask_pyramid=mask_pyramid,
            data_group=feature_group,
            mask_group=mask_group
        )
        assert features.shape[:2] == mask.shape, f"{features.shape[:2]} differs from {mask.shape}"

        # Smooth features a bit
        if smooth_sigma > 0.:
            features = filters.gaussian(features, multichannel=True, sigma=smooth_sigma)

        selected_features.append(features)
        selected_masks.append(mask)

    if mask_features:
        valid_features = [f[m != background_class] for f, m in zip(selected_features, selected_masks)]
    else:
        valid_features = [sf.reshape(-1, sf.shape[-1]) for sf in selected_features]

    valid_lengths = [len(vf) for vf in valid_features]
    valid_features = np.vstack(valid_features)

    print(f"Valid features have shape {valid_features.shape}")

    # Perform KMeans
    from sklearn.cluster import KMeans

    np.random.seed(seed)

    # Reduce to the selected valid components
    ix = np.random.choice(np.arange(len(valid_features)), n_samples)

    print("Fitting KMeans...")
    km = KMeans(n_clusters, n_init=32, max_iter=300, tol=1e-5, random_state=seed, verbose=True)
    km.fit(valid_features[ix])

    # Create masks
    out_path = os.path.join(out_folder, model_name)

    if not os.path.exists(out_path):
        print("Create path", out_path)
        os.makedirs(out_path, exist_ok=True)

    print("Write cluster sections to", out_path)
    for sf, sm, (k, r) in tqdm(zip(selected_features, selected_masks, files_df.iterrows()), total=len(files_df)):

        # Get predictions
        predictions = km.predict(sf[sm != background_class])

        cluster_array = np.zeros(sf.shape[:2], dtype=np.int16)
        cluster_array[sm != background_class] = predictions + 1

        out_file = f"Kmeans_{n_clusters}_s{r.id:04d}.h5"
        out_section = pli.data.Section(image=cluster_array)
        out_section.spacing = r.spacing
        out_section.origin = r.origin
        out_section.modality = out_modality
        out_section.attrs['n_clusters'] = n_clusters

        out_section.to_hdf5(
            os.path.join(out_path, out_file),
            chunk_size=chunk_size,
            compression=compression,
            pyramid=False,
            overwrite=True
        )
        out_section.close_file_handle()

        with h5.File(os.path.join(out_path, out_file), 'r+') as f:
            f.create_dataset("cluster_centers", data=km.cluster_centers_)


if __name__ == "__main__":
    create_clusters()

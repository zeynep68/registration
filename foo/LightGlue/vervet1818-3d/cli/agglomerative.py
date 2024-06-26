import os

import re
import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering

import h5py as h5

import pli
import pli.image as im

import click

from tqdm import tqdm


# Group of the features in the H5 files
cluster_group = "Image"
out_modality = 'Mask'

# Out parameters
chunk_size = 128
compression ='gzip'

seed = 299792458


@click.command()
@click.option("--model_name", type=str)
@click.option("--kmeans_path", default="data/aa/clusters/kmeans_pca_s2_128/", type=str)
@click.option("--out_folder", default='data/aa/clusters', type=str)
@click.option("--n_clusters", default=2, type=int, help="Number of clusters for Agglomerative")
@click.option("--n_max", default=21, type=int, help="Maximum number of clusters vor coloring lower numbers of clusters")
def create_clusters(model_name, kmeans_path, out_folder, n_clusters, n_max):

    p = re.compile('.*s([0-9]{4}).*h5')

    feature_list = []
    cluster_centers = None
    kmeans_path = os.path.join(kmeans_path, model_name)
    for f in sorted(os.listdir(kmeans_path)):
        match = p.match(f)
        if match:
            id = int(match[1])
            cluster_section = pli.data.Section(path=os.path.join(kmeans_path, f))
            cluster_centers = cluster_section.attrs['cluster_centers']
            spacing = cluster_section.spacing
            origin = cluster_section.origin
            cluster_section.close_file_handle()
            feature_list.append({'id': id, 'spacing': spacing, 'origin': origin, 'file': os.path.join(kmeans_path, f)})
    files_df = pd.DataFrame(feature_list)

    def perform_clustering(n_mappings, cluster_centers, n_max):
        ac = AgglomerativeClustering(n_clusters=n_mappings, affinity='euclidean', linkage='ward')
        cluster_mapping = ac.fit_predict(cluster_centers)
        cluster_mapping = (cluster_mapping * n_max) // n_mappings
        cluster_mapping += (n_max - cluster_mapping.max()) // 2

        return cluster_mapping

    cluster_mapping = perform_clustering(n_clusters, cluster_centers, n_max)

    out_path = os.path.join(out_folder, f"agglomerative_{n_clusters}", model_name)
    if not os.path.exists(out_path):
        print("Create path", out_path)
        os.makedirs(out_path, exist_ok=True)

    print("Write cluster sections to", out_path)
    for k, r in tqdm(files_df.iterrows(), total=len(files_df)):
        cluster_section = pli.data.Section(path=r.file)
        section_clusters = cluster_section.image[:]
        test_mask = section_clusters > 0
        section_clusters[test_mask] = cluster_mapping[section_clusters[test_mask] - 1] + 1
        cluster_section.close_file_handle()

        out_file = f"Agglomerative_{n_clusters}_s{r.id:04d}.h5"
        out_section = pli.data.Section(image=section_clusters)
        out_section.spacing = r.spacing
        out_section.origin = r.origin
        out_section.maximum = n_max
        out_section.minimum = 1
        out_section.modality = out_modality
        out_section.attrs['n_mappings'] = n_clusters

        out_section.to_hdf5(
            os.path.join(out_path, out_file),
            chunk_size=chunk_size,
            compression=compression,
            pyramid=False,
            overwrite=True
        )

        out_section.close_file_handle()


if __name__ == "__main__":
    create_clusters()

import math
import re
import os

import pli
import numpy as np
from multiprocessing import Pool

from vervet1818_3d.utils.stats import pli_features, pli_glcm


#def run(ix):
#    results = []
#    for ix in np.arange(ix * chunk_size, (ix + 1) * chunk_size):
#        if ix < trans_ft.shape[0]:
#            ft = pli_features(trans_ft[ix], dir_ft[ix], ret_ft[ix], pli_bins)
#            results.append(ft)
#    return np.vstack(results)

def run_histo(i):
    return pli_features(trans_ft[i], dir_ft[i], ret_ft[i], pli_bins)


def pli2histo(
    trans: np.ndarray,
    dir: np.ndarray,
    ret: np.ndarray,
    pyramid: int,
    n_bins=32,
    n_loaders=8
):
    ps = 2 ** pyramid
    splits = (trans.shape[0] // ps, trans.shape[1] // ps)
    f_shape = (ps * splits[0], ps * splits[1])

    global trans_ft, dir_ft, ret_ft, num_workers, chunk_size, pli_bins

    num_workers = n_loaders
    chunk_size = math.ceil((splits[0] * splits[1]) / num_workers)
    pli_bins = n_bins

    trans_tiled = trans[:f_shape[0], :f_shape[1]].reshape(splits[0], ps, splits[1], ps).transpose(0, 2, 1, 3)
    trans_ft = trans_tiled.reshape(splits[0] * splits[1], -1)
    dir_tiled = dir[:f_shape[0], :f_shape[1]].reshape(splits[0], ps, splits[1], ps).transpose(0, 2, 1, 3)
    dir_ft = dir_tiled.reshape(splits[0] * splits[1], -1)
    ret_tiled = ret[:f_shape[0], :f_shape[1]].reshape(splits[0], ps, splits[1], ps).transpose(0, 2, 1, 3)
    ret_ft = ret_tiled.reshape(splits[0] * splits[1], -1)

    with Pool(n_loaders) as p:
        outputs = p.map(run_histo, range(trans_ft.shape[0]))

    out_ft = np.vstack(outputs).reshape(*splits, -1)

    return out_ft


def run_glcm(i):
    return pli_glcm(trans_ft[i], dir_ft[i], ret_ft[i])


def pli2glcm(
    trans: np.ndarray,
    dir: np.ndarray,
    ret: np.ndarray,
    pyramid: int,
    n_loaders=8
):
    ps = 2 ** pyramid
    splits = (trans.shape[0] // ps, trans.shape[1] // ps)
    f_shape = (ps * splits[0], ps * splits[1])

    global trans_ft, dir_ft, ret_ft, num_workers, chunk_size

    num_workers = n_loaders
    chunk_size = math.ceil((splits[0] * splits[1]) / num_workers)

    trans_tiled = trans[:f_shape[0], :f_shape[1]].reshape(splits[0], ps, splits[1], ps).transpose(0, 2, 1, 3)
    trans_ft = trans_tiled.reshape(splits[0] * splits[1], ps, ps)
    dir_tiled = dir[:f_shape[0], :f_shape[1]].reshape(splits[0], ps, splits[1], ps).transpose(0, 2, 1, 3)
    dir_ft = dir_tiled.reshape(splits[0] * splits[1],  ps, ps)
    ret_tiled = ret[:f_shape[0], :f_shape[1]].reshape(splits[0], ps, splits[1], ps).transpose(0, 2, 1, 3)
    ret_ft = ret_tiled.reshape(splits[0] * splits[1], ps, ps)

    with Pool(n_loaders) as p:
        outputs = p.map(run_glcm, range(trans_ft.shape[0]))

    out_ft = np.vstack(outputs).reshape(*splits, -1)

    return out_ft


def load_dir_files(path, name, regex='.*s([0-9]{4})_.*'):
    import pandas as pd
    
    p = re.compile(regex)

    match_list = []
    for f in sorted(os.listdir(path)):
        matches = p.match(f)
        if matches is not None:
            id = int(matches[1])
            match_list.append({'id': id, f'file_{name}': os.path.join(path, f)})
    return pd.DataFrame(match_list).sort_values('id').reset_index(drop=True)


def read_masked_features(
        feature_path: str,
        mask_path: str,
        mask_pyramid: int,
        data_group: str,
        mask_group: str = 'Image'
):

    import numpy as np

    feature_section = pli.data.Section(feature_path, data_group=data_group)
    mask_section = pli.data.Section(mask_path, data_group=mask_group)

    if mask_pyramid == 0:
        try:
            mask = mask_section.image[:]
        except:
            mask = mask_section.pyramid[mask_pyramid][:]
    else:
        mask = mask_section.pyramid[mask_pyramid][:]
    features = feature_section.image[:].transpose(1, 2, 0).astype(np.float32)

    if features.shape[0] > mask.shape[0]:
        features = features[:-1]
    if features.shape[1] > mask.shape[1]:
        features = features[:, :-1]

    feature_section.close_file_handle()
    mask_section.close_file_handle()

    return features, mask
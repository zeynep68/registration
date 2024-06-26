from scipy import ndimage as ndi
import numpy as np
import h5py as h5

from sklearn.feature_extraction import image
from sklearn.cluster import spectral_clustering

from skimage.transform import resize

from scipy.ndimage import distance_transform_edt

import SimpleITK as sitk


def load_h5(path, dataset):
    with h5.File(path, 'r') as f:
        out = np.array(f[dataset])
    return out


def clean_binary_mask(bin_mask, min_count):
    label_objects, nb_labels = ndi.label(bin_mask)
    sizes = np.bincount(label_objects.ravel())
    mask_sizes = sizes > min_count
    mask_sizes[0] = False
    return mask_sizes[label_objects]


def flip_image(image: sitk.Image, axis=1):
    arr = sitk.GetArrayFromImage(image)
    out_img = sitk.GetImageFromArray(np.flip(arr, axis))
    out_img.SetSpacing(image.GetSpacing())
    return out_img


def mask_array(arr, mask):
    out_arr = arr.copy()
    out_arr[~mask] = 0
    return out_arr


def itk_metric(a, b, metric):
    a_array = sitk.GetArrayFromImage(a)
    b_array = sitk.GetArrayFromImage(b)
    return metric(a_array, b_array)


def to_itk(array, spacing, dtype=np.float32):
    array = array.astype(dtype)
    image = sitk.GetImageFromArray(array)
    if type(spacing) is tuple:
        image.SetSpacing(spacing)
    else:
        image.SetSpacing((spacing, spacing))
    return image


def spectral_mask_clusters(array, mask, k, px, beta, eps):
    # Expect array of gray values. 0 Values are treated as masked

    # Downsize image for better computation
    max_shape = max(array.shape)
    shape = (px * array.shape[0] // max_shape, px * array.shape[1] // max_shape)
    print(f"Downsize blockface to shape {shape}")
    rescaled_array = resize(array, shape)
    if mask is not None:
        # Do not leave any pixel with foreground information unmasked
        # Henve do linear interpolation and select values > 0
        rescaled_mask = resize(mask, shape) > 0.0
    else:
        rescaled_mask = None

    # Create graph
    print(f"Create graph with beta {beta} and eps {eps}")
    graph = image.img_to_graph(rescaled_array, mask=rescaled_mask)
    graph.data = np.exp(-beta * graph.data / graph.data.std()) + eps

    print("Perform spectral clustering")
    labels = spectral_clustering(
        graph, n_clusters=k, assign_labels='kmeans', random_state=42
    )

    if rescaled_mask is not None:
        segments = np.zeros_like(rescaled_array)
        segments[rescaled_mask] = labels + 1
    else:
        segments = labels + 1
    segments = resize(segments, array.shape, order=0)

    return segments


def distance_transform(image, max_dist):
    spacing = image.GetSpacing()
    assert spacing[0] == spacing[1]
    max_dist_mu = max_dist / spacing[0]
    arr = sitk.GetArrayViewFromImage(image)
    out_dist = np.zeros_like(arr)
    for i in np.unique(arr):
        out_dist += np.clip(distance_transform_edt((arr != i)), 0, max_dist_mu) / max_dist_mu
    out_image = sitk.GetImageFromArray(out_dist)
    out_image.SetSpacing(spacing)
    return out_image


def get_bf_segments(bf_gray, bf_mu, k_max, px=100, beta=1, eps=1e-6):
    bf_mask = bf_gray > 0
    out_segments = [to_itk(bf_mask, bf_mu)]
    bf_gray_01 = bf_gray / 255
    out_bf = [to_itk(mask_array(bf_gray_01, bf_mask), bf_mu)]
    if k_max <= 1:
        print(f"Generate segments k=1")
        return out_segments
    for k in np.arange(2, k_max + 1):
        print(f"Generate segments k={k}")
        segments = spectral_mask_clusters(bf_gray_01, bf_mask, k, px, beta, eps)
        out_segments += [to_itk(segments == (i + 1), bf_mu) for i in range(k)]
        out_bf += [to_itk(mask_array(bf_gray_01, segments == (i + 1)), bf_mu) for i in range(k)]
    return out_segments, out_bf


def get_mask_segments(mask, mu, ntrans=None, min_size=1e4):
    mask_filled = ~clean_binary_mask(~mask, min_size)
    label_objects, nb_labels = ndi.label(mask_filled)
    out_segments = []
    out_ntrans = None if ntrans is None else []
    for i in range(nb_labels):
        label_mask = label_objects == (i + 1)
        if np.sum(label_mask) >= min_size:
            if ntrans is not None:
                ntrans_segment = 1 - np.clip(ntrans, 0, 1)
                ntrans_segment[~label_mask] = 0
                out_ntrans.append(to_itk(ntrans_segment, mu))
            out_segments.append(to_itk(label_mask, mu))
    return out_segments, out_ntrans

import numpy as np
from sklearn.decomposition import PCA


# Apply sobel filter for direciton on unit circle and use abs difference
def dir_sobel_filter(dir):
    from scipy import ndimage

    cmplx_dir = np.cos(2 * dir) + 1j * np.sin(2 * dir)

    cmplx_sobel_h = ndimage.sobel(cmplx_dir, 0)
    cmplx_sobel_v = ndimage.sobel(cmplx_dir, 1)

    cmplx_sobel = np.abs(cmplx_sobel_h + cmplx_sobel_v) / 12 # Makes sure sobel filtered results are in [0, 1]

    return cmplx_sobel


def center_dir(direction):
    """

    :param direction: Flattened direction of shape (N) samples
    :return: Direction centered to main principal component axis
    """
    # Project direction to unit circle
    im_dir = np.sin(direction)
    real_dir = np.cos(direction)
    cmplx_dir = np.stack([im_dir, real_dir], axis=-1)
    cmplx_mean = np.mean(cmplx_dir, axis=0)

    # Get main axis and secondary axis (sorted)
    scatter_matrix = cmplx_dir.T @ cmplx_dir / len(cmplx_dir)
    eig_val, eig_vec = np.linalg.eig(scatter_matrix)
    principal_axis = eig_vec[:, np.argsort(eig_val)[-1]]

    if np.linalg.norm(principal_axis - cmplx_mean) > np.linalg.norm(-principal_axis - cmplx_mean):
        principal_axis = -principal_axis

    # Project direction onto eigen basis
    delta_phi = np.arctan2(*principal_axis)

    center_dir = (direction - delta_phi) % np.pi

    return center_dir


# TODO https://docs.opencv.org/3.4/df/dd2/tutorial_py_surf_intro.html

def pli_features(trans, dir, ret, n_bins=16, norm_trans=1.0):
    from scipy import ndimage

    trans_flat = trans.flatten() / norm_trans
    dir_flat = dir.flatten()
    ret_flat = ret.flatten()

    c_dir = center_dir(dir_flat)

    trans_hist, _ = np.histogram(trans_flat, bins=n_bins, range=(0, 1))
    dir_hist, _ = np.histogram(c_dir, bins=n_bins, range=(0, np.pi))
    ret_hist, _ = np.histogram(ret_flat, bins=n_bins, range=(0, 1))

    trans_sobel = ndimage.sobel(trans / norm_trans).flatten()
    dir_sobel = dir_sobel_filter(dir).flatten()
    ret_sobel = ndimage.sobel(ret).flatten()

    trans_sobel_hist, _ = np.histogram(trans_sobel, bins=n_bins, range=(-1, 1))
    dir_sobel_hist, _ = np.histogram(dir_sobel, bins=n_bins, range=(0, 2))
    ret_sobel_hist, _ = np.histogram(ret_sobel, bins=n_bins, range=(-1, 1))

    return np.hstack([trans_hist, ret_hist, dir_hist,
                      trans_sobel_hist, ret_sobel_hist, dir_sobel_hist])


def glcm_features(arr, levels=32, distances=[1, 2, 4], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4]):
    from skimage.feature import greycomatrix, greycoprops

    # Compute GLCM
    glcm = greycomatrix(arr, distances=distances, angles=angles, levels=levels, symmetric=True, normed=True)

    # Extract texture features (you can choose different properties)
    contrast = np.mean(greycoprops(glcm, prop='contrast'), axis=1) # 3 features
    correlation = np.mean(greycoprops(glcm, prop='correlation'), axis=1)
    energy = np.mean(greycoprops(glcm, prop='energy'), axis=1)
    homogeneity =np.mean(greycoprops(glcm, prop='homogeneity'), axis=1)
    
    return np.concatenate([contrast, correlation, energy, homogeneity]) # 12 features


def pli_glcm(trans, dir, ret, levels=32, norm_trans=1.0, distances=[1, 2, 4], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4]):

    dim_1 = np.clip((levels * trans) / norm_trans, 0, levels - 1).astype(np.uint8)
    dim_2 = np.clip(levels * ret, 0, levels - 1).astype(np.uint8)
    dim_3 = np.clip(levels * dir_sobel_filter(dir), 0, levels - 1).astype(np.uint8)

    features = np.hstack([
        glcm_features(dim, levels=levels, distances=distances, angles=angles) for dim in [dim_1, dim_2, dim_3]
    ])  # 36 features

    return features


def pca_n_transform(X, pca, n):
    """
    Only perform PCA on the first n components

    :param X: array
    :param pca: fitted PCA
    :param n: n components
    :return: 
    """
    if pca.mean_ is not None:
        X = X[...] - pca.mean_
    return np.dot(X, pca.components_[:n].T)


def norm_log_likelihood(x: np.ndarray, mu: float, std: float):
    return np.sum(-0.5 * ((x - mu) / std) ** 2 - np.log(np.sqrt(2 * np.pi) * std))


def profile_likelihood(L: int, pca: PCA):
    """
    Profile likelihood

    https://www.sciencedirect.com/science/article/abs/pii/S0167947305002343

    :param L: number of used PCA components
    :param pca: fitted PCA
    :return: profile log likelihood
    """
    mu_1 = np.mean(pca.explained_variance_[:L]) if L > 0 else 0
    mu_2 = np.mean(pca.explained_variance_[L:]) if L < pca.n_components_ else 0
    var_1 = np.sum((pca.explained_variance_[:L] - mu_1) ** 2) if L > 0 else 0
    var_2 = np.sum((pca.explained_variance_[L:] - mu_2) ** 2) if L < pca.n_components_ else 0
    var = (var_1 + var_2) / pca.n_components_
    std = np.sqrt(var)

    log_norm_1 = norm_log_likelihood(pca.explained_variance_[:L], mu_1, std) if L > 0 else 0
    log_norm_2 = norm_log_likelihood(pca.explained_variance_[L:], mu_2, std) if L < pca.n_components_ else 0

    return log_norm_1 + log_norm_2
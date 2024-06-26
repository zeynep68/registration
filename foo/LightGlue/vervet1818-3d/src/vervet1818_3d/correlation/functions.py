import matplotlib.pyplot as plt
import numpy as np


def load_data(target_file, cortex_file):
    import nibabel as nib
    import numpy as np
    import h5py as h5

    target_nifti = nib.load(target_file)
    target_spacing = tuple(np.diag(target_nifti.affine)[:3])
    target_volume = np.array(target_nifti.dataobj)[:, ::-1, ::-1]

    with h5.File(cortex_file, 'r') as f:
        cortex_volume = f['volume'][:][::-1, ::-1, ::-1]
        cortex_spacing = tuple(f['volume'].attrs['spacing'])

    return target_volume, target_spacing, cortex_volume, cortex_spacing


def load_features(features_dir, feature_group, zero_section, cortex_volume):

    import h5py as h5
    import os
    import re
    from tqdm import tqdm

    feature_maps = {}

    for f in tqdm(sorted(os.listdir(features_dir))):
        with h5.File(os.path.join(features_dir, f), 'r') as ft_f:
            ft = ft_f[feature_group]
            feature_spacing = ft.attrs['spacing']
            index = int(re.search(".*(s\d{4}).*", f)[1][1:]) - zero_section
            fm = ft[:].transpose(1, 2, 0)
            if fm.shape[:2] != (cortex_volume.shape[0], cortex_volume.shape[2]):
                feature_maps[index] = fm[:-1, :-1]
            else:
                feature_maps[index] = fm

    return feature_maps


def aggregate_data(indices, subsample_count, seed, cortex_volume, feature_maps, target_volume, gm_class=2):
    features_list = []
    target_list = []

    for i in indices:
        cortex_mask = cortex_volume[:, i, :] == gm_class
        features = feature_maps[i][cortex_mask]
        ce = target_volume[:, i, :][cortex_mask]

        features_list += list(features)
        target_list += list(ce)

    # Subsample training examples
    np.random.seed(seed)
    train_ix = np.random.choice(np.arange(len(features_list)), size=subsample_count, replace=False)
    feature_array = np.array([features_list[i] for i in train_ix])
    depth_array = np.array([target_list[i] for i in train_ix])

    return feature_array, depth_array


def r_squared(pred, target, target_range, clip: bool):
    if clip:
        pred_clip = np.clip(pred, *target_range)
    else:
        pred_clip = pred
    u = np.sum((target - pred_clip) ** 2)
    v = np.sum((target - np.mean(target)) ** 2)
    r_sq = 1 - (u / v)

    return r_sq


def plot_regression(target, pred, target_range=(0, 1), vis_range=(-0.5, 1.5), r_sq=None, title="Cortical depth", subsample=10_000, file=None):
    
    linewidth = .5
    
    fig, ax = plt.subplots(figsize=(1.75, 1.75))

    ax.tick_params(axis='both', which='both', width=linewidth, length=2.)

    ax.set_xlabel("True", fontsize=7)
    ax.grid(linewidth=linewidth)
    ax.set_ylabel("Prediction", fontsize=7)
    ax.set_aspect('equal', 'box')
    plt.xticks(np.linspace(target_range[0], target_range[1], 6), fontsize=5)
    plt.yticks(np.linspace(target_range[0], target_range[1], 6), fontsize=5)
    ax.set(xlim=(vis_range[0], vis_range[1]), ylim=(vis_range[0], vis_range[1]))
    ax.set_title(title, fontsize=7, fontweight='bold')

    if len(target) > subsample:
        np.random.seed(299_792_458)
        ix = np.random.choice(np.arange(len(target)), subsample, replace=False)
    else:
        ix = ...

    ax.scatter(
        target[ix],
        pred[ix],
        s=1.,
        marker='.',
        zorder=7,
        alpha=0.5,
        edgecolors='none',
    )
    ax.plot([target_range[0], target_range[1]], [target_range[0], target_range[1]], linewidth=linewidth, color='red', zorder=8)

    for spine in ax.spines.values():
        spine.set_linewidth(linewidth)  # Set the width to 2, adjust as needed

    # ax.vlines([target_range[0], target_range[1]], vis_range[0], vis_range[1], color='black', linestyles='-', zorder=9)
    # ax.hlines([target_range[0], target_range[1]], vis_range[0], vis_range[1], color='black', linestyles='--', zorder=10)

    if r_sq is not None:
        plt.text(
            (target_range[1] - target_range[0]) / 2 + target_range[0],
            target_range[0],
            f"R²={r_sq:.02f}",
            horizontalalignment='center',
            verticalalignment='bottom',
            fontsize=7,
            zorder=11,
        )

    if file:
        try:
            fig.savefig(file + ".png", dpi=450, bbox_inches='tight')
            fig.savefig(file + ".svg", bbox_inches='tight')
        except:
            print(f"Could not write file {file}. Is it write protected?")
    plt.show()


def linear_evaluation(train_features, train_target, test_features, test_target, target_range, vis_range,
                      name, model, title, subsample=10_000):

    from sklearn import preprocessing
    from sklearn.linear_model import LinearRegression

    # Standardize features
    print("Scale")
    scaler = preprocessing.StandardScaler().fit(train_features)
    train_features_scaled = scaler.transform(train_features)
    test_features_scaled = scaler.transform(test_features)

    # Train regression model
    print("Fit")
    reg = LinearRegression().fit(train_features_scaled, train_target)

    # Evaluate overfitting
    print("Train & Train:")
    train_pred = reg.predict(train_features_scaled)

    train_train_r_sq = r_squared(train_pred, train_target, target_range=target_range, clip=False)
    print(f"R² train: {train_train_r_sq:.3g}")
    plot_regression(train_target, train_pred, target_range=target_range, vis_range=vis_range, subsample=subsample,
                    r_sq=train_train_r_sq, title=title, file=f"doc/correlation/{name}_{model}_train_train")

    # Evaluate prediction
    print("Train & Test:")

    test_pred = reg.predict(test_features_scaled)

    train_test_r_sq = r_squared(test_pred, test_target, target_range=target_range, clip=False)
    print(f"R² test: {train_test_r_sq:.3g}")
    plot_regression(test_target, test_pred, target_range=target_range, vis_range=vis_range, subsample=subsample,
                    r_sq=train_test_r_sq, title=title, file=f"doc/correlation/{name}_{model}_train_test")

    # Evaluate test test prediction
    print("Test & Test:")
    print("Scale")
    scaler = preprocessing.StandardScaler().fit(test_features)
    test_features_scaled = scaler.transform(test_features)

    # Train regression model
    print("Fit")
    reg = LinearRegression().fit(test_features_scaled, test_target)
    test_pred = reg.predict(test_features_scaled)

    test_test_r_sq = r_squared(test_pred, test_target, target_range=target_range, clip=False)
    print(f"R² tes testt: {test_test_r_sq:.3g}")
    plot_regression(test_target, test_pred, target_range=target_range, vis_range=vis_range, subsample=subsample,
                    r_sq=test_test_r_sq, title=title, file=f"doc/correlation/{name}_{model}_test_test")

    return {'train_train_r2': train_train_r_sq, 'train_test_r2': train_test_r_sq, 'test_test_r2': test_test_r_sq}


def ridge_evaluation(train_features, train_target, test_features, test_target, target_range, vis_range,
                      name, model, title, subsample=10_000):

    from sklearn import preprocessing
    from sklearn.linear_model import Ridge

    # Standardize features
    print("Scale")
    scaler = preprocessing.StandardScaler().fit(train_features)
    train_features_scaled = scaler.transform(train_features)
    test_features_scaled = scaler.transform(test_features)

    # Train regression model
    print("Fit")
    reg = Ridge(alpha=1e4).fit(train_features_scaled, train_target)

    # Evaluate overfitting
    print("Train & Train:")
    train_pred = reg.predict(train_features_scaled)

    train_train_r_sq = r_squared(train_pred, train_target, target_range=target_range, clip=True)
    print(f"R² train: {train_train_r_sq:.3g}")
    plot_regression(train_target, train_pred, target_range=target_range, vis_range=vis_range, subsample=subsample,
                    r_sq=train_train_r_sq, title=title, file=f"doc/ridge/{name}_{model}_train_train")

    # Evaluate prediction
    print("Train & Test:")

    test_pred = reg.predict(test_features_scaled)

    train_test_r_sq = r_squared(test_pred, test_target, target_range=target_range, clip=True)
    print(f"R² test: {train_test_r_sq:.3g}")
    plot_regression(test_target, test_pred, target_range=target_range, vis_range=vis_range, subsample=subsample,
                    r_sq=train_test_r_sq, title=title, file=f"doc/ridge/{name}_{model}_train_test")

    # Evaluate test test prediction
    print("Test & Test:")
    print("Scale")
    scaler = preprocessing.StandardScaler().fit(test_features)
    test_features_scaled = scaler.transform(test_features)

    # Train regression model
    print("Fit")
    reg = Ridge(alpha=1e4).fit(test_features_scaled, test_target)
    test_pred = reg.predict(test_features_scaled)

    test_test_r_sq = r_squared(test_pred, test_target, target_range=target_range, clip=True)
    print(f"R² tes testt: {test_test_r_sq:.3g}")
    plot_regression(test_target, test_pred, target_range=target_range, vis_range=vis_range, subsample=subsample,
                    r_sq=test_test_r_sq, title=title, file=f"doc/ridge/{name}_{model}_test_test")

    return {'train_train_r2': train_train_r_sq, 'train_test_r2': train_test_r_sq, 'test_test_r2': test_test_r_sq}

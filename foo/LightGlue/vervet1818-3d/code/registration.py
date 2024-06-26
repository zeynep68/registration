# import the necessary packages
from collections import namedtuple

import numpy as np
import imutils
import SimpleITK as sitk

import pli
import pli.image as im

import utils


Transform = namedtuple('Transform', ('transform', 'ntrans_ix', 'bf_ix', 'flip', 'metric'))


def transform_image(image, trans, reference, interpolator=sitk.sitkLinear):
    # TODO Store spacings and shape, not reference iamge
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference)
    resampler.SetInterpolator(interpolator)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(trans)

    return resampler.Execute(image)


def exhaustive_euler(fixed, moving, fixed_mask, moving_mask, n_angles=180, flip=False):
    moving_flipped = utils.flip_image(moving) if flip else moving
    moving_mask_flipped = utils.flip_image(moving_mask) if flip else moving_mask
    initial_transform = sitk.CenteredTransformInitializer(
        fixed_mask, moving_mask_flipped, sitk.Euler2DTransform(), sitk.CenteredTransformInitializerFilter.MOMENTS
    )

    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMattesMutualInformation()
    R.SetInterpolator(sitk.sitkLinear)
    R.SetOptimizerAsExhaustive([n_angles // 2, 0, 0])
    R.SetOptimizerScales([2.0 * np.pi / n_angles, 1.0, 1.0])
    R.SetInitialTransform(initial_transform, inPlace=True)
    R.SetMetricMovingMask(moving_mask_flipped)

    R.Execute(fixed, moving_flipped)
    return initial_transform, R.GetMetricValue()


def gradient_optimization(fixed, moving, fixed_mask, moving_mask, init_transform, n_iterations, lr, flip=False):
    moving_mask = utils.flip_image(moving_mask) if flip else moving_mask
    moving = utils.flip_image(moving) if flip else moving

    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMattesMutualInformation()
    R.SetMetricMovingMask(moving_mask)
    R.SetOptimizerAsGradientDescent(learningRate=lr, numberOfIterations=n_iterations, convergenceMinimumValue=1e-12)
    R.SetOptimizerScalesFromPhysicalShift()
    # R.SetMovingInitialTransform(init_transform)
    R.SetInitialTransform(init_transform, inPlace=False)

    final_transform = R.Execute(fixed, moving)
    return final_transform, R.GetMetricValue()


def exhaustive_best_fit(bf_segments, bf_masks, ntrans_segments, ntrans_masks, n_samples, n_iterations, lr, allow_flip=False):
    transforms = []
    best_transforms = []
    i = 0
    for t_ix, (t_s, t_m) in enumerate(zip(ntrans_segments, ntrans_masks)):
        best_transform = None
        metric = np.finfo(float).max
        for bf_ix, (bf_s, bf_m) in enumerate(zip(bf_segments, bf_masks)):
            flip_l = [False, True] if allow_flip else [False]
            for flip in flip_l:
                i_transform, _ = exhaustive_euler(bf_s, t_s, fixed_mask=bf_m, moving_mask=t_m,
                                                         n_angles=n_samples, flip=flip)
                f_transform, c_metric = gradient_optimization(bf_s, t_s, bf_m, t_m,
                                                              i_transform, n_iterations, lr,
                                                              flip=flip)
                print(f"{i} \tmask {t_ix} \tbf {bf_ix} \tflip {flip} \tmetric {c_metric}")
                i += 1

                transform = Transform(f_transform, t_ix, bf_ix, flip, c_metric)
                if c_metric < metric:
                    metric = c_metric
                    best_transform = transform
                transforms.append(transform)

        best_transforms.append(best_transform)

    return best_transforms, transforms

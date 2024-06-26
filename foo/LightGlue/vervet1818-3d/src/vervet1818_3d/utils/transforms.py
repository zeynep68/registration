import numpy as np
import SimpleITK as sitk
from skimage.util.shape import view_as_windows


def downscale_image(itk_image: sitk.SimpleITK.Image, factor: float, interpolator: int):
    field_size = itk_image.GetSize()
    field_spacing = itk_image.GetSpacing()
    field_dimension = itk_image.GetDimension()

    output_size = [int(field_size[d] / factor) for d in range(field_dimension)]
    output_spacing = [field_spacing[d] * factor for d in range(field_dimension)]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(output_spacing)
    resample.SetSize(output_size)
    resample.SetInterpolator(interpolator)

    return resample.Execute(itk_image)


def load_disp_field(file):
    transform = sitk.DisplacementFieldTransform(sitk.ReadTransform(file))
    field_spacing = tuple(transform.GetFixedParameters()[4:6])

    return transform, field_spacing


def index2coord(ix, spacing):
    return tuple(reversed([(i + 0.5) * s for i, s in zip(ix, spacing)]))


def warp_coord(coord, transform):
    return transform.TransformPoint(coord)


def coord2ix(coord, spacing):
    return tuple(reversed([int(c // s) for c, s in zip(coord, spacing)]))
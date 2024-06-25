import numpy as np
import json
import h5py
import SimpleITK as sitk


def get_same_physical_size(bf_img, trans_img, bf_mask_img, trans_mask_img, bf_spacing, trans_spacing):
	return


def to_itk(array, spacing, dtype=np.float32):
    array = array.astype(dtype)
    image = sitk.GetImageFromArray(array)
    if type(spacing) is tuple:
        #spacing += (1.0,)
        image.SetSpacing(spacing)
    else:
        image.SetSpacing((spacing, spacing))
    return image


def process_string(section_no):
	n = len(section_no)
	
	out = "0" * n
	out += section_no
	return out


def load_images(bf_pyramid_level, trans_pyramid_level, section):
	# load two images : one blockface + one transmittance
	data = json.load(open('pli_paths.json'))
	out = process_string(section)
	print(section, out)

	bf_path = data['blockface'][section]
	trans_path = data['transmittance'][section]

	bf_path = bf_path.replace("/p/data1/", "/home/zeynepboztoprak/p/data1/")
	trans_path = trans_path.replace("/p/data1/", "/home/zeynepboztoprak/p/data1/")

	bf_fd = h5py.File(bf_path, 'r')
	trans_fd = h5py.File(trans_path, 'r')

	trans_mask_path = f"/p/data1/pli/DB_data/PE-2021-00981-H_ImageData/Brain_Part_00/Section_{out}/PM/Complete/Mask/PE-2021-00981-H_00_s{out}_PM_Complete_Mask_Stitched_Flat_v000.h5"
	trans_mask_path = trans_mask_path.replace("/p/data1/", "/home/zeynepboztoprak/p/data1/")
	trans_mask_fd = h5py.File(trans_mask_path, 'r')

	bf_mask_path = f"/p/data1/pli/DB_data/PE-2021-00981-H_ImageData/Brain_Part_00/Section_{out}/BF/Mask/PE-2021-00981-H_00_s{out}_BF_Mask_Registered_Flat_v000.h5"
	bf_mask_path = bf_mask_path.replace("/p/data1/", "/home/zeynepboztoprak/p/data1/")
	bf_mask_fd = h5py.File(bf_mask_path, 'r')

	if bf_pyramid_level is None:
		bf_img = bf_fd['Image'][:]
		bf_mask_img = bf_mask_fd['Image'][:]
	else:
		bf_img = bf_fd['pyramid'][bf_pyramid_level][:]
		bf_mask_img = bf_mask_fd['pyramid'][bf_pyramid_level][:]

	trans_img = trans_fd['pyramid'][trans_pyramid_level][:]
	trans_mask_img = trans_mask_fd['pyramid'][trans_pyramid_level][:]

	return (bf_img, bf_mask_img), (trans_img, trans_mask_img)

	

def register_masks(fixed_mask, moving_mask):
    """
    Registers the moving mask to the fixed mask and saves the registered mask.

    Parameters:
    fixed_mask_path (str): Path to the fixed mask image.
    moving_mask_path (str): Path to the moving mask image.
    output_registered_mask_path (str): Path to save the registered mask image.
    """
    # Initialisierung der Registrierungsmethode
    registration_method = sitk.ImageRegistrationMethod()

    # Festlegen der Metrik (z.B. Mattes Mutual Information)
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)

    # Anwenden der Masken (beide Masken sind nun die eigentlichen "Bilder" für die Registrierung)
    registration_method.SetMetricFixedMask(fixed_mask)
    registration_method.SetMetricMovingMask(moving_mask)

    # Konfiguration des Optimizers und Interpolators
    registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100)
    registration_method.SetInterpolator(sitk.sitkNearestNeighbor)

    # Initiale Transformation (Identität)
    initial_transform = sitk.CenteredTransformInitializer(fixed_mask, 
                                                          moving_mask, 
                                                          sitk.Euler2DTransform(), 
                                                          sitk.CenteredTransformInitializerFilter.GEOMETRY)
    registration_method.SetInitialTransform(initial_transform, inPlace=False)

    # Ausführen der Registrierung
    final_transform = registration_method.Execute(fixed_mask, moving_mask)

    # Anwenden der Transformation auf die bewegliche Maske
    resampled_moving_mask = sitk.Resample(moving_mask, fixed_mask, final_transform, sitk.sitkNearestNeighbor, 0.0, moving_mask.GetPixelID())

    return resampled_moving_mask

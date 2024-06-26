import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import center_of_mass


def preprocess_transmittance_mask(mask, center=None, angle=None):
	"""
		center: Rotation along this center coordinates.
	"""
	values = np.unique(mask)
	
	for v in values:
		assert v in [0., 100., 200., 255.], 'Wrong intensity value in mask!!'

	mask[mask > 0.] = 255  # set to white - gray / white matter segmentation is combined

	if angle is not None:
		mask = rotate_img(mask, angle)

	mask[mask < 255.] = 0.  # set to black

	return mask


def rotate_img(img, angle, center=None):
    if center is None:
        center = (img.shape[1] // 2, img.shape[0] // 2)

    # Compute the rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Calculate the new bounding dimensions of the image after rotation
    angle_rad = np.deg2rad(angle)
    cos_val = np.abs(np.cos(angle_rad))
    sin_val = np.abs(np.sin(angle_rad))
    
    new_width = int((img.shape[0] * sin_val) + (img.shape[1] * cos_val))
    new_height = int((img.shape[0] * cos_val) + (img.shape[1] * sin_val))

    # Adjust the rotation matrix to account for translation
    M[0, 2] += (new_width / 2) - center[0]
    M[1, 2] += (new_height / 2) - center[1]

    # Perform the rotation
    rotated_img = cv2.warpAffine(img, M, (new_width, new_height))
    return rotated_img


def compute_scaling_factor(fixed_um_per_px, moving_um_per_px, fixed_pyramid_lvl, moving_pyramid_lvl):
	return (moving_um_per_px * 2**moving_pyramid_lvl) / (fixed_um_per_px * 2**fixed_pyramid_lvl)


def rescale(moving, scaling_factor, padding_shape=None, interpolation=None, padding_value=None):
	if interpolation is None:
		interpolation = cv2.INTER_CUBIC

	moving = cv2.resize(moving, (0,0), fx=scaling_factor, fy=scaling_factor, interpolation=interpolation)
	
	if padding_shape is not None:
		moving = pad_img(moving, padding_shape, padding_value=padding_value)

	return moving


def pad_img(image, target_shape, padding_value=0):
	original_height, original_width = image.shape[:2]
	target_height, target_width = target_shape
    
	padding_top = max((target_height - original_height) // 2, 0)
	padding_bottom = max(target_height - original_height - padding_top, 0)
	padding_left = max((target_width - original_width) // 2, 0)	
	padding_right = max(target_width - original_width - padding_left, 0)

	return cv2.copyMakeBorder(image, padding_top,
							  padding_bottom, 
      						  padding_left, padding_right, 
         					  borderType=cv2.BORDER_CONSTANT, 
        					  value=padding_value)


def blend_images(img1, img2, alpha=0.5, dpi=250, figsize=(10,5)):
	plt.figure(dpi=dpi, figsize=figsize)
	plt.imshow(img1, cmap='Reds', alpha=alpha)
	plt.imshow(img2, cmap='Greens', alpha=alpha)
	plt.show()


def normalize_pixel_values(img, max_val=1.):  # [0, max_val]
	numerator = img - np.min(img) 
	denominator = np.max(img) - np.min(img)

	return (numerator / denominator) * max_val


def compute_center_of_mass(img):
	return center_of_mass(img)


def draw_red_dot(img, coord, title='Image with center of mass'):
	plt.imshow(img, cmap='gray')
	plt.scatter(coord[1], coord[0], color='red', s=100)
	plt.title(title)
	plt.show()


def compute_translation_vector(fixed_com, moving_com):
	return (fixed_com[1] - moving_com[1] , fixed_com[0] - moving_com[0])
	

def translate_image(img, translation_vec):
	M = np.float32([[1, 0, translation_vec[0]], [0, 1, translation_vec[1]]])

	return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

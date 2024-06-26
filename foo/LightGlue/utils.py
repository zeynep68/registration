import cv2
import h5py
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def blend_images(img1, img2, alpha=0.5, dpi=250, figsize=(10,5)):
	plt.figure(dpi=dpi, figsize=figsize)
	plt.imshow(img1, cmap='Reds', alpha=alpha)
	plt.imshow(img2, cmap='Greens', alpha=alpha)
	plt.show()


def convert_to(img, dtype=np.float32):
	if dtype == np.float32:
		return img.astype(dtype)
	elif dtype == np.uint8:
		if np.max(img) <= 1.:
			img = img * 255.
		return img.astype(np.uint8)	
	else:
		print('dtype not supported!')


def normalize_pixel_values(img, max_val=1.):  # [0, max_val]
	numerator = img - np.min(img) 
	denominator = np.max(img) - np.min(img)

	return (numerator / denominator) * max_val


def plot_two_imgs(img1, img2):
	fig, axes = plt.subplots(1, 2, figsize=(10, 5))

	# Display the first image
	axes[0].imshow(img1, cmap='gray')
	axes[0].axis('off')  # Hide axes

	# Display the second image
	axes[1].imshow(img2, cmap='gray')
	axes[1].axis('off')  # Hide axes

	plt.show()


def rgb_to_grayscale(img):
	return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def resize_img(img, target_shape, interpolation=cv2.INTER_CUBIC):
	""" Not wanted! Object deformation / distortion
	"""
	return cv2.resize(img, target_shape, interpolation=interpolation)


def compute_scale_factor(original_shape, target_shape):
    """
    Compute the scale factor needed to resize an image from the original shape
    to the target shape while maintaining the aspect ratio.
    
    Parameters:
        original_shape (tuple): The shape (height, width) of the original image.
        target_shape (tuple): The desired shape (height, width) to which the image should be resized.
        
    Returns:
        float: The scale factor by which to resize the image.
    """
    original_height, original_width = original_shape
    target_height, target_width = target_shape
    
    # Compute scale factors for each dimension
    scale_factor_height = target_height / original_height
    scale_factor_width = target_width / original_width
    
    # Choose the smaller scale factor to maintain the aspect ratio
    scale_factor = min(scale_factor_height, scale_factor_width)
    
    return scale_factor


def rescale_img(img, target_shape, interpolation=cv2.INTER_CUBIC):
	""" Rescales an image by a scale factor using PIL, maintaining aspect ratio."""
	scale_factor = compute_scale_factor(img.shape, target_shape)

	width = int(img.shape[1] * scale_factor)
	height = int(img.shape[0] * scale_factor)
	new_dim = (width, height)

	return cv2.resize(img, new_dim, interpolation=interpolation)


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


def select_appropriate_pyramid_level(img1_path, img2_path=None, img2_shape=None):
	""" img1_path: Path to hdf5 file for trans, ret or dir image
		img2_path: Path to hdf5 file for blockface image

		Choose the largest pyramid level where img1 is smaller than
		the blockface image in all dimensions.
	 """
	if img2_shape is not None:
		blockface_shape = img2_shape
	else:
		blockface_shape = h5py.File(img2_path, 'r')['Image'].shape

	for k in reversed(h5py.File(img1_path, 'r')['pyramid']):
		shape = h5py.File(img1_path, 'r')['pyramid'][k].shape	

		if (shape[0] >= blockface_shape[0]) and (shape[1] >= blockface_shape[1]):
			return k


def get_bounding_box(img: np.ndarray, mask):
    """
    Returns:
    - bbox_coords: Coordinates of the bounding box (x, y, w, h).
    """
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        # Get the bounding box for the largest contour
        contour = max(contours, key=cv2.contourArea)

        x, y, w, h = cv2.boundingRect(contour)

        bbox_coords = (x, y, w, h)
        
        return bbox_coords
    else:
        print("No contours found in the mask.")
        return None


def rotate_image1(img, angle):
    # Get image dimensions
    (h, w) = img.shape[:2]
    
    # Calculate the center of the image
    center = (w // 2, h // 2)
    
    # Perform the rotation
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(img, M, (w, h))
    
    return rotated_image


def rotate_image(img, angle):
    # Get image dimensions
    (h, w) = img.shape[:2]
    
    # Calculate the center of the image
    center = (w // 2, h // 2)
    
    # Calculate the rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Calculate the sine and cosine of the rotation angle
    abs_cos = abs(M[0, 0])
    abs_sin = abs(M[0, 1])
    
    # Calculate the new dimensions of the bounding box
    new_w = int(h * abs_sin + w * abs_cos)
    new_h = int(h * abs_cos + w * abs_sin)
    
    # Adjust the rotation matrix to account for the translation
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]
    
    # Perform the rotation with the new dimensions
    rotated_image = cv2.warpAffine(img, M, (new_w, new_h))
    
    return rotated_image

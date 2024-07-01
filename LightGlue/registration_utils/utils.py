import cv2
import numpy as np


def rgb_to_gray(img):
	return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


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


def invert_img(img):
	return np.max(img) - img


def normalize_pixel_values(img, max_val=1.):  # [0, max_val]
	numerator = img - np.min(img) 
	denominator = np.max(img) - np.min(img)

	return (numerator / denominator) * max_val

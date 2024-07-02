import cv2
import torch
import numpy as np
from registration_utils.utils import rgb_to_gray, invert_img, normalize_pixel_values


def compute_scaling_factor(fixed_um_per_px, moving_um_per_px, fixed_pyramid_lvl, moving_pyramid_lvl):
	return (moving_um_per_px * 2**moving_pyramid_lvl) / (fixed_um_per_px * 2**fixed_pyramid_lvl)


def rescale_img(moving, scaling_factor, padding_shape=None, interpolation=None):
	if interpolation is None:
		interpolation = cv2.INTER_CUBIC

	moving = cv2.resize(moving, (0,0), fx=scaling_factor, fy=scaling_factor, interpolation=interpolation)
	
	return moving


class Constants:
	def __init__(self, bf_pyramid_lvl, trans_pyramid_lvl):	
		self.BF_PYRAMID_LVL = bf_pyramid_lvl
		self.TRANS_PYRAMID_LVL = trans_pyramid_lvl

		self.int_bf_pyramid_lvl = int(bf_pyramid_lvl)
		self.int_trans_pyramid_lvl = int(trans_pyramid_lvl)
	
		self.BF_UM_PER_PX = 32.18880081176758
		self.TRANS_UM_PER_PX = 1.33

		self.scaling_factor = compute_scaling_factor(fixed_um_per_px=self.BF_UM_PER_PX, 
							     moving_um_per_px=self.TRANS_UM_PER_PX, 
							     fixed_pyramid_lvl=self.int_bf_pyramid_lvl, 
							     moving_pyramid_lvl=self.int_trans_pyramid_lvl)

	def register_section(self, data, normalize_value=1.0, use_trans_mask=False, return_torch_tensor=True):
		bf_img = data['Blockface'][:]
		# to 1 channel dim
		bf_img = rgb_to_gray(bf_img)

		trans_img = data['Transmittance'][:]
		bf_mask_img = data['BF-Mask'][:] 
		trans_mask_img = data['TRANS-Mask'][:] 

		# convert to same physical resolution
		trans_img = rescale_img(trans_img, self.scaling_factor, bf_img.shape)
		trans_mask_img = rescale_img(trans_mask_img, self.scaling_factor, bf_img.shape)
		
		# set values to either 0 or 255

		bf_mask_img[bf_mask_img < 127] = 0 
		bf_mask_img[bf_mask_img >= 127] = 255

		trans_mask_img[trans_mask_img < 127] = 0 
		trans_mask_img[trans_mask_img >= 127] = 255

		bf_img = np.where(bf_mask_img, bf_img, normalize_value)
		if use_trans_mask:
			trans_img = np.where(trans_mask_img, trans_mask_img, normalize_value)

		inv_trans_img = invert_img(trans_img)
		inv_bf_img = invert_img(bf_img)

		bf_img = normalize_pixel_values(bf_img, max_val=normalize_value)
		trans_img = normalize_pixel_values(trans_img, max_val=normalize_value)
		inv_bf_img = normalize_pixel_values(inv_bf_img, max_val=normalize_value)
		inv_trans_img = normalize_pixel_values(inv_trans_img, max_val=normalize_value)

		bf_img[bf_img < 0] = 0
		trans_img[trans_img < 0] = 0
		inv_bf_img[inv_bf_img < 0] = 0
		inv_trans_img[inv_trans_img < 0] = 0

		if return_torch_tensor:
			return self.to_torch([bf_img, trans_img, bf_mask_img, trans_mask_img, inv_bf_img, inv_trans_img])

		return bf_img, trans_img, bf_mask_img, trans_mask_img, inv_bf_img, inv_trans_img

	def to_torch(self, elements):
		return [torch.tensor(e).cuda().float().unsqueeze(0).unsqueeze(0) for e in elements]




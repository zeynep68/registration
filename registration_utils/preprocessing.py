import cv2
import h5py
import torch
import numpy as np
from registration_utils.utils import invert_img, normalize_pixel_values


class DataPreparation:
	def __init__(self, fixed_pyramid_lvl, moving_pyramid_lvl, 
              	 fixed_um_per_px=32.18880081176758, moving_um_per_px=1.33):	
     
		self.compute_scaling(moving_um_per_px=moving_um_per_px,
                             fixed_um_per_px=fixed_um_per_px,
							 moving_pyramid_lvl=moving_pyramid_lvl,
        					 fixed_pyramid_lvl=fixed_pyramid_lvl)

	def compute_scaling(self, moving_um_per_px, fixed_um_per_px, 
                        moving_pyramid_lvl, fixed_pyramid_lvl): 
		moving = moving_um_per_px * 2**moving_pyramid_lvl  
		fixed = fixed_um_per_px * 2**fixed_pyramid_lvl
  
		self.scaling = moving / fixed
  
	def to_grayscale(self, img):
		return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	def rescale_img(self, moving, interpolation=None):
		if interpolation is None:
			interpolation = cv2.INTER_CUBIC
    
		return cv2.resize(moving, (0,0), fx=self.scaling_factor, 
                    	  fy=self.scaling_factor, interpolation=interpolation)

	def load_images(self, data, section_id, prefix="/home/zeynepboztoprak/p/data1/"):
		bf_mask_path = data['BF-Mask'][section_id][0]
		bf_mask_path = bf_mask_path.replace("/p/data1/", prefix)

		mod_mask_path = data['TRANS-Mask'][section_id][0]
		mod_mask_path = mod_mask_path.replace("/p/data1/", prefix)

		bf_path = data['Blockface'][section_id][0]
		bf_path = bf_path.replace("/p/data1/", prefix)

		trans_path = data['Transmittance'][section_id][0]
		trans_path = trans_path.replace("/p/data1/", prefix)
	
		bf_mask = h5py.File(bf_mask_path, 'r')['pyramid'][self.BF_PYRAMID_LVL]
		mod_mask = h5py.File(mod_mask_path, 'r')['pyramid'][self.TRANS_PYRAMID_LVL]
		bf = h5py.File(bf_path, 'r')['pyramid'][self.BF_PYRAMID_LVL]

		trans = h5py.File(trans_path, 'r')['pyramid'][self.TRANS_PYRAMID_LVL]
  
		return {'Transmittance': trans, 'BF-Mask': bf_mask, 'Blockface': bf, 'TRANS-Mask': mod_mask}

	def preprocess(self, moving, fixed, moving_mask, fixed_mask, maxval=1.0):
		fixed = self.to_grayscale(fixed)	
		fixed = np.where(fixed_mask, fixed, maxval)
  
		fixed_mask = self.fix_mask(fixed_mask)
  
		moving = self.rescale_img(moving, fixed.shape)
		moving = self.invert_img(moving)
  
		moving_mask = self.rescale_img(moving_mask, fixed.shape)
		moving_mask = self.fix_mask(moving_mask)

		fixed = self.normalization(fixed, max_val=maxval)
		moving = self.normalization(moving, max_val=maxval)

		return moving, fixed, moving_mask, fixed_mask 

	def fix_mask(self, mask, threshold=99):
		mask[mask < threshold] = 0.
		mask[mask >= threshold] = 255.
  
		return mask

	def invert_img(img):
		return np.max(img) - img

	def normalization(self, img, maxval=1.):  # [0, max_val]
		img = ( (img - np.min(img) ) / (np.max(img) - np.min(img) ) ) * maxval
  
		img[img < 0] = 0
  
		return img

	def to_torch(self, elements):  # lightglue expects [b,c,h,w]
		return [torch.tensor(e).cuda().float().unsqueeze(0).unsqueeze(0) for e in elements]  




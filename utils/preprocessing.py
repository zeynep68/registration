import cv2
import numpy as np

class DataCuration:
	def __init__(self, fixed_pyramid_lvl, moving_pyramid_lvl,
		fixed_um_per_px=32.18880081176758, moving_um_per_px=1.33):
     
		self.compute_scaling(moving_um_per_px=moving_um_per_px,
							 fixed_um_per_px=fixed_um_per_px,
							 moving_pyramid_lvl=moving_pyramid_lvl,
							 fixed_pyramid_lvl=fixed_pyramid_lvl)


	def compute_scaling(self, moving_um_per_px, fixed_um_per_px, moving_pyramid_lvl, fixed_pyramid_lvl):
		moving = moving_um_per_px * 2**int(moving_pyramid_lvl)
		fixed = fixed_um_per_px * 2**int(fixed_pyramid_lvl)

		self.scaling = moving / fixed

	def to_grayscale(self, img):
		return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	def rescale_img(self, moving, interpolation=None):
		if interpolation is None:
			interpolation = cv2.INTER_CUBIC

		return cv2.resize(moving, (0,0), fx=self.scaling, fy=self.scaling, interpolation=interpolation)

	def prepare(self, fixed, fixed_mask, moving, moving_mask, maxval=1.0):
		fixed = self.to_grayscale(fixed)	
		fixed = np.where(fixed_mask, fixed, maxval)
  
		fixed_mask = self.fix_mask(fixed_mask)
  
		moving = self.rescale_img(moving)
		moving = self.invert_img(moving)
  
		moving_mask = self.rescale_img(moving_mask)
		moving_mask = self.fix_mask(moving_mask)

		fixed = self.normalization(fixed, maxval=maxval)
		moving = self.normalization(moving, maxval=maxval)

		return (fixed, fixed_mask), (moving, moving_mask)

	def fix_mask(self, mask, threshold=99):
		mask[mask < threshold] = 0.
		mask[mask >= threshold] = 255.

		return mask

	def invert_img(self, img):
		return np.max(img) - img

	def normalization(self, img, maxval=1.):  # [0, max_val]
		img = ( (img - np.min(img) ) / (np.max(img) - np.min(img) ) ) * maxval
		img[img < 0] = 0
  
		return img

	def to_torch(self, elements):  # lightglue expects [b,c,h,w]
		import torch
  
		return [torch.tensor(e).cuda().float().unsqueeze(0).unsqueeze(0) for e in elements]  


def load_images(data, section_id, fixed_pyramid_lvl, moving_pyramid_lvl):
	import h5py

	def sub_fn(data, section_id, key, pyramid_lvl='00',
			   prefix="/home/zeynepboztoprak/p/data1/"):
		path = data[key][section_id][0]
		path = path.replace("/p/data1/", prefix)

		return h5py.File(path, 'r')['pyramid'][pyramid_lvl][:]

	fixed = sub_fn(data, section_id, pyramid_lvl=fixed_pyramid_lvl, key='Blockface')
	fixed_mask = sub_fn(data, section_id, pyramid_lvl=fixed_pyramid_lvl, key='BF-Mask')

	moving = sub_fn(data, section_id, pyramid_lvl=moving_pyramid_lvl, key='Transmittance')
	moving_mask = sub_fn(data, section_id, pyramid_lvl=moving_pyramid_lvl, key='TRANS-Mask')


	return (fixed, fixed_mask), (moving, moving_mask)
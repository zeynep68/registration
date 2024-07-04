import h5py
import json
import numpy as np


def rescale_img(img, interpolation=None):
		if interpolation is None:
			interpolation = cv2.INTER_CUBIC

		return cv2.resize(img, (0,0), fx=self.scaling, fy=self.scaling, interpolation=interpolation)


if __name__ == "__main__":
	PATH = "/home/zeynepboztoprak/code/registration/LightGlue/registration_utils/pli_big_brain_paths.json"
	data = json.load(open(PATH, "r"))

	SECTION_ID = "920"

	blockface = data['Blockface'][SECTION_ID][0]
	transmittance = data['Transmittance'][SECTION_ID][0]
	
	blockface = blockface.replace("/p/data1/", "/home/zeynepboztoprak/p/data1/")
	transmittance = transmittance.replace("/p/data1/", "/home/zeynepboztoprak/p/data1/")

	blockface = h5py.File(blockface, 'r')['pyramid']
	transmittance = h5py.File(transmittance, 'r')['pyramid']

	blockface_um_per_px = 32.18880081176758
	transmittance_um_per_px = 1.33

	#print('Blockface shapes')
	#for k in blockface.keys():
#		numerator = blockface_um_per_px * 2**int(k)
#		denominator = transmittance_um_per_px * 2**int(k)
#
#		scaling_factor = numerator / denominator
#
#		print(blockface[k].shape, np.array(blockface[k].shape)*scaling_factor)

	print('Transmittance shapes')
	for i in range(10):
		print('blockface pyramid lvl:', i)
		for k in transmittance.keys():
			denominator = blockface_um_per_px * 2**int(i)
			numerator = transmittance_um_per_px * 2**int(k)

			scaling_factor = numerator / denominator

			print(transmittance[k].shape, np.array(transmittance[k].shape)*scaling_factor)
		print()
		print()
	
	print('Blockface shapes')
	for k in blockface.keys():
		print(k,blockface[k].shape)


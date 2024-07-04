import os
import cv2
import json
import h5py
import matplotlib.pyplot as plt


def blend_images(img1, img2, s1, s2, prefix='before_', alpha=0.2, 
				 dpi=250, figsize=(10,5)):
	plt.figure(dpi=dpi, figsize=figsize)
	plt.imshow(img1, cmap='Reds', alpha=alpha)
	plt.imshow(img2, cmap='Greens', alpha=alpha)
	plt.savefig(f"registered/{prefix}" + s1 + "_" + s2 + ".png")
	plt.close()


def register(fixed, moving):
	return


def format_section_string(s):
	return (4 - len(s)) * "0" + s


def get_transformations_between_two_sections(all_transformation_files, f1, f2):
	""" Get all transformation matrices between any two consecutive sections 
		that lie between sections s1 and s2.
	"""
	filtered = [f for f in all_transformation_files if f >= f1]
	filtered = [f for f in filtered if f <= f2]

	return filtered


def main(s_prev, s_next, blockfaces, path_to_transformations):  # in ascending order: s_prev < s_nextA
	"""
	Parameters:
		s_prev (str): Previous section within the stack.
		s_next (str): Current section that should be aligned with s_prev.
		blockfaces (dict): The path to the corresponding blockface image is stored for each section_id.
		PATH (str): Directory containing the transformations of two successive blockface images.
	"""
	img1 = h5py.File(blockfaces[s_prev][0].replace("/p/data1/", "/home/zeynepboztoprak/p/data1/"), 'r')['Image'][:]
	#img2 = h5py.File(blockfaces[s_next][0].replace("/p/data1/", "/home/zeynepboztoprak/p/data1/"), 'r')['Image'][:]
	img2 = h5py.File('/home/zeynepboztoprak/p/data1/pli/DB_data/PE-2021-00981-H_ImageData/Brain_Part_00/Section_0321/BF/Raw/PE-2021-00981-H_00_s0321_BF_Raw_Registered_Flat_v000.h5','r')['Image'][:]

	s_prev = format_section_string(s_prev)
	s_next = format_section_string(s_next)
	
	f1 = f"PE-2021-00981-H_PLI-BigBrain-li_s{s_prev}.Composite.h5"
	f2 = f"PE-2021-00981-H_PLI-BigBrain-li_s{s_next}.Composite.h5"

	filtered = get_transformations_between_two_sections(os.listdir(path_to_transformations), f1, f2)
	filtered.sort()	
	
	#t1 = h5py.File(path_to_transformations + filtered[0], 'r')['TransformGroup']['0']['TransformFixedParameters']
	t1 = h5py.File(path_to_transformations + filtered[0], 'r')['TransformGroup']['0']['TransformParameters']
	t1 = t1[:].reshape(2,-1)
	#print(t1.keys())
	blend_images(img1, img2, s_prev, s_next)
	translated_img = cv2.warpAffine(img1, t1, img1.shape[:2])
	blend_images(translated_img, img2, s_prev, s_next, prefix='after_')

	

	
	
	return


if __name__ == "__main__":
	brain = "PE-2021-00981-H"

	if brain == "PE-2021-00981-H":
		PATH_TO_TRANSFORMATIONS = "/home/zeynepboztoprak/p/data1/pli/Private/schober1/bfreg/PE-2021-00981-H/parameters/reg04-planar-correction/"
		PATH_TO_BLOCKFACE = "/home/zeynepboztoprak/code/registration/pli-to-bf/metadata/PE_2021_00981_H.json"
	else:
		PATH = "/home/zeynepboztoprak/p/data1/pli/Private/schober1/bfreg/PE-2020-00691-H/parameters/reg06-cropped/"

	
	blockfaces = json.load(open(PATH_TO_BLOCKFACE, 'r'))['blockface']

	main(s_prev="320", s_next="321", blockfaces=blockfaces, path_to_transformations=PATH_TO_TRANSFORMATIONS)

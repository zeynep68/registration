import os
import json
import h5py
import matplotlib.pyplot as plt
from registration_utils.preprocessing import DataPreparation

os.environ['QT_QPA_PLATFORM'] = 'offscreen'

def blend_images(img1, img2, alpha=0.5, dpi=250, figsize=(10,5), path=None):
	plt.figure(dpi=dpi, figsize=figsize)
	plt.imshow(img1, cmap='Reds', alpha=alpha)
	plt.imshow(img2, cmap='Greens', alpha=alpha)
	if path is not None:
		plt.savefig(path)


def save_images(img1, img2, title1='', title2='', output_path=''):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

	# Display the first image
    ax1.imshow(img1, cmap='gray')
    ax1.set_title(title1)
    ax1.axis('off')  # Hide axes

    # Display the second image
    ax2.imshow(img2, cmap='gray')
    ax2.set_title(title2)
    ax2.axis('off')  # Hide axes

    # Save the figure
    plt.savefig(output_path, bbox_inches='tight')
    plt.close(fig)


def main(bf_img, trans_img, bf_mask_img, trans_mask_img, inv_bf_img, inv_trans_img, section):
	save_images(bf_img, bf_mask_img, title1='Blockface', title2='Blockface mask', output_path="./results/" + str(section) + "/images/bf_mask.png")
	save_images(trans_img, trans_mask_img, title1='Transmittance', title2='Transmittance mask', output_path="./results/" + str(section) + "/images/trans_mask.png")
	save_images(bf_img, inv_bf_img, title1='Blockface', title2='Blockface inverted', output_path="./results/" + str(section) + "/images/bf_inv.png")
	save_images(trans_img, inv_trans_img, title1='Transmittance', title2='Transmittance inverted', output_path="./results/" + str(section) + "/images/trans_inv.png")

	torch_bf_img, torch_trans_img, torch_bf_mask_img, torch_trans_mask_img, torch_inv_bf_img, torch_inv_trans_img = constants.to_torch([bf_img, trans_img, bf_mask_img, trans_mask_img, inv_bf_img, inv_trans_img])

	print(np.max(inv_trans_img), np.min(inv_trans_img))
	print(np.max(inv_bf_img), np.min(inv_bf_img))
	print(np.max(trans_img), np.min(trans_img))
	print(np.max(bf_img), np.min(bf_img))

	blend_images(bf_img, trans_img, path="./results/" + str(section) + "/images/blended_bf_trans.png")
	
	###############################################################
	#### Pre-registration
	###############################################################
	translation_vec, rotation_angle, transformed_img = pre_register(bf_mask_img, trans_mask_img, trans_img)

	blend_images(bf_img, transformed_img, path="./results/" + str(section) + "/images/blended_bf_transformed.png")
	put_everything_together(torch_inv_trans_img, torch_bf_img, path="./results/" + str(section) + "/lightglue-registration/ODE_transformation_loss.png")
	

	

	
	
	
	
	
	return


if __name__ == "__main__":
	config = {'data_path': "/home/zeynepboztoprak/code/registration/LightGlue/registration_utils/pli_big_brain_paths.json",
			  'BF_PYRAMID_LVL': '00',
			  'TRANS_PYRAMID_LVL': '06',
			  'section': "920"}

	data = json.load(open(config['data_path'], "r"))
	print(data.keys())

	preprocessor = DataPreparation(bf_pyramid_lvl=config['BF_PYRAMID_LVL'], trans_pyramid_lvl=config['TRANS_PYRAMID_LVL'])

	data = preprocessor.load_images(data, config['section'])

	bf_img, trans_img, bf_mask_img, trans_mask_img, inv_bf_img, inv_trans_img = preprocessor.preprocess(data, return_torch_tensor=False)

	main(bf_img, trans_img, bf_mask_img, trans_mask_img, inv_bf_img, inv_trans_img, config['section'])

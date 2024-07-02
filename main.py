import os
import json
import h5py
import matplotlib.pyplot as plt
from registration_utils.preprocessing import DataCuration, load_images 

os.environ['QT_QPA_PLATFORM'] = 'offscreen'

def blend_images(img1, img2, alpha=0.5, dpi=250, figsize=(10,5), path=None):
	plt.figure(dpi=dpi, figsize=figsize)
	plt.imshow(img1, cmap='Reds', alpha=alpha)
	plt.imshow(img2, cmap='Greens', alpha=alpha)
	if path is not None:
		plt.savefig(path)


def save_images(img1, img2, title1='', title2='', path=''):
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
    plt.savefig(path, bbox_inches='tight')
    plt.close(fig)


def main(fixed, fixed_mask, moving, moving_mask, section_id, path):
	save_images(fixed, fixed_mask, title1='Blockface', title2='Blockface mask', path=path + "images/blockface.png")
 	save_images(moving, moving_mask, title1='Transmittance', title2='Transmittance mask', path=path + "images/transmittance.png")
	
	dc = DataCuration()
 
	torch_fixed, torch_fixed_mask, torch_moving, torch_moving_mask = dc.to_torch([fixed, fixed_mask, 
                                                                               	  moving, moving_mask])

	blend_images(fixed, moving, path=path + "/images/before_pre_reg.png")
	
	###############################################################
	#### Pre-registration
	###############################################################
	#translation_vec, rotation_angle, transformed_img = pre_register(bf_mask_img, trans_mask_img, trans_img)

	#blend_images(bf_img, transformed_img, path="./results/" + str(section) + "/images/blended_bf_transformed.png")
	#put_everything_together(torch_inv_trans_img, torch_bf_img, path="./results/" + str(section) + "/#lightglue-registration/ODE_transformation_loss.png")


if __name__ == "__main__":
    PATH = "/home/zeynepboztoprak/code/registration/LightGlue/registration_utils/pli_big_brain_paths.json"
    data = json.load(open(PATH, "r"))
    
    SECTION_ID = 920
    
    (fixed, fixed_mask), (moving, moving_mask) = load_images(data, section_id=SECTION_ID)
    
    main(fixed, fixed_mask, moving, moving_mask, SECTION_ID, path="./results/" + str(section) + "/")

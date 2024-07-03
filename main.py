import cv2
import json
import matplotlib.pyplot as plt
from utils.preprocessing import DataCuration, load_images 
from utils.pre_registration import pre_register
from utils.lightglue_reg import put_everything_together


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
 

def blend_images(img1, img2, alpha=0.5, dpi=250, figsize=(10,5), path=None):
	plt.figure(dpi=dpi, figsize=figsize)
	plt.imshow(img1, cmap='Reds', alpha=alpha)
	plt.imshow(img2, cmap='Greens', alpha=alpha)
	if path is not None:
		plt.savefig(path)


def save_images(img1, img2, title1='', title2='', path='', keep_axis=False):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

	# Display the first image
    ax1.imshow(img1, cmap='gray')
    ax1.set_title(title1)
    if not keep_axis:
    	ax1.axis('off')  # Hide axes

    # Display the second image
    ax2.imshow(img2, cmap='gray')
    ax2.set_title(title2)
    if not keep_axis:
    	ax2.axis('off')  # Hide axes

    # Save the figure
    plt.savefig(path, bbox_inches='tight')
    plt.close(fig)


def main(dc, fixed, fixed_mask, moving, moving_mask, section_id, path):
    print('... Before rescaling ...')
    print('Blockface:',fixed.shape, fixed_mask.shape)
    print('Transmittance:', moving.shape, moving_mask.shape)
    
    print('... After rescaling ...')
    (fixed, fixed_mask), (moving, moving_mask) = dc.prepare(fixed, fixed_mask, moving, moving_mask)
    print('Blockface:',fixed.shape, fixed_mask.shape)
    print('Transmittance:', moving.shape, moving_mask.shape)
    
    print('---'*15)
    print(f"Memory usage of blockface array: {fixed.nbytes / (1024 ** 2):.2f} MB")
    print(f"Memory usage of transmittance array: {moving.nbytes / (1024 ** 2):.2f} MB")
    print('---'*15)
    
    save_images(fixed, fixed_mask, title1='Blockface', title2='Blockface mask', path=path + "images/blockface.png")
    save_images(moving, moving_mask, title1='Transmittance', title2='Transmittance mask', path=path + "images/transmittance.png")
    save_images(pad_img(moving, fixed.shape), fixed, title1='Transmittance', title2='Blockface', path=path + "images/both_modalities.png", keep_axis=True)

    blend_images(fixed, moving, path=path + "/images/before_pre_reg.png")
	
	###############################################################
	#### Pre-registration
   	###############################################################
    translation, rotation_angle, transformed = pre_register(fixed_mask, moving_mask, moving, path=path + "images/")
    
    torch_fixed, torch_moving, torch_transformed = dc.to_torch([fixed, moving, transformed])
    
    blend_images(fixed, transformed, path=path + "/images/after_pre_reg.png")
    
    print('---'*15)
    print('Lightglue registration ...')
    put_everything_together(torch_transformed, torch_fixed, path=path + "images/")


if __name__ == "__main__":
    PATH = "/home/zeynepboztoprak/code/registration/metadata/pli_big_brain_paths.json"
    data = json.load(open(PATH, "r"))
    
    SECTION_ID = "920"

    FIXED_PYRAMID_LVL = '00'
    MOVING_PYRAMID_LVL = '06'

    dc = DataCuration(fixed_pyramid_lvl=FIXED_PYRAMID_LVL, moving_pyramid_lvl=MOVING_PYRAMID_LVL)
    
    print('---'*15 + '\n' + 'Data loading ...')
    (fixed, fixed_mask), (moving, moving_mask) = load_images(data, section_id=SECTION_ID,
                                                             fixed_pyramid_lvl=FIXED_PYRAMID_LVL,
                                                             moving_pyramid_lvl=MOVING_PYRAMID_LVL)
    print('Completed!' + '\n' + '---'*15)
    main(dc, fixed, fixed_mask, moving, moving_mask, SECTION_ID, path="./results/PE-2021-00981-H/" + str(SECTION_ID) + "/")

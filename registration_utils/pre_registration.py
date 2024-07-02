import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.ndimage import center_of_mass


def compute_center_of_mass(img):
	return center_of_mass(img)


def draw_red_dot(img, coord, path, title='Image with center of mass'):
    plt.imshow(img, cmap='gray')
    plt.scatter(coord[1], coord[0], color='red', s=100)
    plt.title(title)
    plt.savefig(path + "COM_" + title + ".png")
    plt.close()
    

def compute_translation_vector(fixed_com, moving_com):
	return (fixed_com[1] - moving_com[1] , fixed_com[0] - moving_com[0])
	

def translate_image(img, translation_vec):
	M = np.float32([[1, 0, translation_vec[0]], [0, 1, translation_vec[1]]])

	return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))


def pre_register(mask_fixed, mask_moving, moving, verbose=True, return_translated_img=True, path=None):
	com1 = compute_center_of_mass(mask_fixed)	
	com2 = compute_center_of_mass(mask_moving)	

	translation_vec = compute_translation_vector(fixed_com=com1, moving_com=com2)

	if verbose:
		draw_red_dot(mask_fixed, com1, path, title='Blockface')
		draw_red_dot(mask_moving, com2, path, title='Transmittance')

		print('COM fixed img:', com1)
		print('COM moving img:', com2)
		print('translation vec:', translation_vec)

	translated = translate_image(moving, translation_vec)
    
	rotation_angle = compute_rotation_angle(mask_fixed, mask_moving)

	rotated_translated = rotate_image(translated, rotation_angle)
	
	if verbose:
		print('rotation angle:', rotation_angle)
	
	return translation_vec, rotation_angle, rotated_translated
	

def extract_coordinates(mask):
    """Extract the coordinates of the non-zero pixels in the mask."""
    coords = np.column_stack(np.nonzero(mask))
    return coords


def compute_pca_angle(coords):
    """Compute the angle of the first principal component with respect to the x-axis."""
    pca = PCA(n_components=2)
    pca.fit(coords)
    # Principal component (the first eigenvector)
    pc1 = pca.components_[0]
    # Angle of the principal component with respect to the x-axis
    angle = np.arctan2(pc1[1], pc1[0])
    return angle


def compute_rotation_angle(mask_fixed, mask_moving):
    coords1 = extract_coordinates(mask_fixed)
    coords2 = extract_coordinates(mask_moving)
    
    # Compute PCA angles
    angle1 = compute_pca_angle(coords1)
    angle2 = compute_pca_angle(coords2)

	# Compute the difference in angles
    rotation_angle = np.rad2deg(angle1 - angle2)
    return rotation_angle	


def plot_masks_with_pca(mask, angle, title=''):
    """Plot the mask with the principal component."""
    plt.imshow(mask, cmap='gray')
    plt.title(title)
    center = center_of_mass(mask)
    length = max(mask.shape) / 2
    plt.arrow(center[1], center[0], length * np.cos(angle), length * np.sin(angle), color='red', head_width=5)
    plt.show()


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


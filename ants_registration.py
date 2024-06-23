import ants
import numpy as np
import matplotlib.pyplot as plt


def ants_registration(fixed: np.ndarray, moving: np.ndarray, type_of_transformation):
	fixed = ants.from_numpy(fixed)
	moving = ants.from_numpy(moving)
		
	# Perform intensity-based registration
	registration = ants.registration(fixed=fixed, moving=moving, type_of_transform=type_of_transformation)
	
	# Get the transformed (registered) image	
	registered = registration['warpedmovout']
	
	# Apply forward transformation to the moving image
	forward_transform = registration['fwdtransforms']
	moving_transformed = ants.apply_transforms(fixed=fixed, moving=moving, transformlist=forward_transform)

	# Apply inverse transformation to the registered image
	inv_transform = registration['invtransforms']
	registered_inv_transformed = ants.apply_transforms(fixed=moving, moving=registered, transformlist=inv_transform)

	return {'registered': registered, 
			'forward_transform': forward_transform, 
			'inv_transform': inv_transform, 
			'moving_transformed': moving_transformed, 
			'registered_inv_transformed': registered_inv_transformed}


def plot_images(fixed, moving, registered, forward_mapped, inverse_mapped):
    fig, axes = plt.subplots(1, 6, figsize=(30, 6))
    
    axes[0].imshow(fixed, cmap='gray')
    axes[0].set_title('Fixed Image')
    axes[0].axis('off')
    
    axes[1].imshow(moving, cmap='gray')
    axes[1].set_title('Moving Image')
    axes[1].axis('off')
    
    axes[2].imshow(registered.numpy(), cmap='gray')
    axes[2].set_title('Registered Image')
    axes[2].axis('off')
    
    axes[3].imshow(forward_mapped.numpy(), cmap='gray')
    axes[3].set_title('Forward Mapped Image')
    axes[3].axis('off')
    
    axes[4].imshow(inverse_mapped.numpy(), cmap='gray')
    axes[4].set_title('Inverse Mapped Image')
    axes[4].axis('off')

    axes[5].imshow(fixed, cmap='Reds', alpha=0.5)
    axes[5].imshow(registered.numpy(), cmap='Greens', alpha=0.5)
    axes[5].set_title('Blended Image After Registration')
    axes[5].axis('off')
    
    plt.show()


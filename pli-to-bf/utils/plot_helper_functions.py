import matplotlib.pyplot as plt


def blend_images(img1, img2, alpha=0.5, 
				 dpi=250, figsize=(10,5)):
	plt.figure(dpi=dpi, figsize=figsize)
	plt.imshow(img1, cmap='Reds', alpha=alpha)
	plt.imshow(img2, cmap='Greens', alpha=alpha)
	plt.show()


def plot_two_imgs(img1, img2):
	fig, axes = plt.subplots(1, 2, figsize=(10, 5))

	# Display the first image
	axes[0].imshow(img1, cmap='gray')
	axes[0].axis('off')  # Hide axes

	# Display the second image
	axes[1].imshow(img2, cmap='gray')
	axes[1].axis('off')  # Hide axes

	plt.show()

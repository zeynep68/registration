import torch
import matplotlib.pyplot as plt 
from lightglue import LightGlue, SuperPoint, DISK, SIFT, ALIKED, DoGHardNet
from lightglue.utils import rbd
from hidra.transformations import ODEPolyAffineTransform


def apply_matching(img1, img2, max_num_keypoints=2048, extractor_str='superpoint', path=None):
    if extractor_str == 'superpoint':
        extractor = SuperPoint(max_num_keypoints=max_num_keypoints).eval().cuda()  # load the extractor
    elif extractor_str == 'disk':
        extractor = DISK(max_num_keypoints=max_num_keypoints).eval().cuda()  # load the extractor
        
    matcher = LightGlue(features=extractor_str).eval().cuda()  # load the matcher
    
    feats0 = extractor.extract(img1)  # auto-resize the image, disable with resize=None
    feats1 = extractor.extract(img2)

    # match the features
    matches01 = matcher({'image0': feats0, 'image1': feats1})
    feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]  # remove batch dimension
    matches = matches01['matches']  # indices with shape (K,2)
    points0 = feats0['keypoints'][matches[..., 0]]  # coordinates in image #0, shape (K,2)
    points1 = feats1['keypoints'][matches[..., 1]]  # coordinates in image #1, shape (K,2)
    
    print(points0.shape, points1.shape)
    
    fig, axes = plt.subplots(1, 2, sharey=True, dpi=250)
    
    axes[0].imshow(img1.cpu().squeeze(), cmap="gray")
    axes[0].scatter(points0.cpu()[:, 0], points0.cpu()[:, 1], s=1)
    
    axes[1].imshow(img2.cpu().squeeze(), cmap="gray")
    axes[1].scatter(points1.cpu()[:, 0], points1.cpu()[:, 1], s=1)
    plt.savefig(path + "lightglue_features.png")
    plt.close()
    
    return points0, points1


def put_everything_together(img1, img2, feature_extractor='superpoint', max_num_keypoints=2048, k=20, path=None):
    p0, p1 = apply_matching(img1, img2, extractor_str=feature_extractor, max_num_keypoints=max_num_keypoints, path=path)

    p0 = p0.flip(1)
    p1 = p1.flip(1)
    
    transformation = ODEPolyAffineTransform()
    loss = transformation.fit(torch.cat((p0, p1), axis=1), n_components=k)
    
    plt.plot(loss)
    plt.savefig(path + "loss.png")
    plt.close()
    
    with torch.no_grad():
        moving_transformed = transformation.transform_image(torch.as_tensor(img1.squeeze()).to(torch.float32),(0, 0) + img2.squeeze().shape,output_res=1.0)
    
    plt.figure(dpi=250)
    plt.imshow(img2.cpu().squeeze(), cmap="Reds", alpha=0.5)
    plt.imshow(moving_transformed.cpu().numpy(), cmap="Greens", alpha=0.5)
    plt.savefig(path + "lightglue_transformed.png")
    plt.close()
    
    fig, axes = plt.subplots(1, 3, figsize=(10, 7))  # figsize can be adjusted based on your image dimensions#
    
    # Display the first image
    axes[0].imshow(img2.cpu().squeeze(), cmap='gray')
    axes[0].axis('off')  # Turn off axis numbering and ticks
    axes[0].set_title('Blockface')
    
    
    # Display the second image
    axes[1].imshow(img1.cpu().squeeze(), cmap='gray')
    axes[1].axis('off')
    axes[1].set_title('Transmittance')
    
    # Display the second image
    axes[2].imshow(moving_transformed.cpu().numpy(), cmap='gray')
    axes[2].axis('off')
    axes[2].set_title('Registered Transmittance')
    
    # Show the plot
    plt.tight_layout()
    plt.savefig(path + "final_result.png")
    plt.close()
    return moving_transformed
    

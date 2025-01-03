import os
import matplotlib
import matplotlib.pyplot as plt
import cv2
import kornia as K
import kornia.feature as KF
import numpy as np
import torch

from kornia_moons.feature import *
import torchvision
from kornia_moons.viz import draw_LAF_matches
import torchvision.transforms.functional as F
import torchvision.transforms as T
from torchvision.utils import save_image
import rasterio
from PIL import Image
from lightglue.utils import match_pair


def load_torch_image(fname, w, h):
    img = K.image_to_tensor(cv2.resize(cv2.imread(fname), (w, h)), False).float() / 255
    img = K.color.bgr_to_rgb(img)
    return img


def resize_torch_crop(tensor_image, w, h):
    tensor_image_resized = F.resize(tensor_image, (h, w))
    return tensor_image_resized


def match_lightglue_crop(tensor_img1, tensor_img2, new_w, new_h, matcher, extractor, old_size, start_w, start_h, device):
    cut_img1 = resize_torch_crop(tensor_img1, new_w, new_h)
    cut_img2 = resize_torch_crop(tensor_img2, new_w, new_h)

    if torch.cuda.is_available():
        cut_img1 = cut_img1.cuda(device).float()
        cut_img2 = cut_img2.cuda(device).float()

    with torch.no_grad():
            feats0, feats1, matches01 = match_pair(
                extractor, matcher,
                K.color.rgb_to_grayscale(cut_img1).to(device),
                K.color.rgb_to_grayscale(cut_img2).to(device),
                device)
    
    kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]

    m_kpts0, m_kpts1 = kpts0[matches[..., 0]].cpu().numpy(), kpts1[matches[..., 1]].cpu().numpy()
    kpts0, kpts1 = kpts0.cpu().numpy(), kpts1.cpu().numpy()
    reposition = lambda x : x / (new_w / old_size[0], new_h / old_size[1]) + (start_w, start_h)
    # return (m_kpts0, m_kpts1, kpts0, kpts1)
    return (reposition(m_kpts0),
            reposition(m_kpts1),
            reposition(kpts0),
            reposition(kpts1))


def split_image(image, n, save_crops=False, saving_path=None):
    _, c, h, w = image.size()
    if n % 2 != 0:
        raise ValueError("N should be an even number")

    step_h = h // (n // 2)
    step_w = w // (n // 2)

    cropped_images = {}
    for i in range(n // 2):
        for j in range(n // 2):
            # Calculate start and end positions with adjustments
            start_h = i * step_h
            end_h = (i + 1) * step_h
            start_w = j * step_w
            end_w = (j + 1) * step_w

            # Adjust for top and bottom edges
            if i == 0:  # Top edge
                end_h += 6
            elif i == n // 2 - 1:  # Bottom edge
                start_h -= 6
            else:  # Middle crops
                start_h -= 3
                end_h += 3

            # Adjust for left and right edges
            if j == 0:  # Left edge
                end_w += 6
            elif j == n // 2 - 1:  # Right edge
                start_w -= 6
            else:  # Middle crops
                start_w -= 3
                end_w += 3

            # Ensure boundaries are valid
            start_h = max(0, start_h)
            end_h = min(h, end_h)
            start_w = max(0, start_w)
            end_w = min(w, end_w)

            # Crop the image
            crop_height = end_h - start_h
            crop_width = end_w - start_w
            cropped = image.narrow(2, start_h, crop_height).narrow(3, start_w, crop_width)

            # Save coordinates and crop
            coordinates = {'start_h': start_h, 'end_h': end_h, 'start_w': start_w, 'end_w': end_w}
            key = f'cropped_{i}_{j}'
            cropped_images[key] = {'coordinates': coordinates, 'image': cropped}

            # Save the crop as an image file
            if save_crops:
                if not saving_path:
                    raise ValueError("Saving path must be provided if save_crops is True")
                filename = f"_{start_h}_{end_h}_{start_w}_{end_w}.png"
                filepath = os.path.join(saving_path, filename)
                save_image(cropped, filepath)

    return cropped_images


def save_image(image, path):
    image_pil = F.to_pil_image(image)
    image_pil.save(path)


def match_loftr_crop(tensor_img1, tensor_img2, new_w, new_h, matcher, old_size, start_w, start_h, device):
    cut_img1 = resize_torch_crop(tensor_img1, new_w, new_h)
    cut_img2 = resize_torch_crop(tensor_img2, new_w, new_h)

    input_dict = {'image0': K.color.rgb_to_grayscale(cut_img1),
                  'image1': K.color.rgb_to_grayscale(cut_img2)}

    # Move input tensors to CUDA if available and cast to torch.cuda.FloatTensor
    if torch.cuda.is_available():
        cut_img1 = cut_img1.cuda(device).float()
        cut_img2 = cut_img2.cuda(device).float()
        input_dict = {key: value.cuda(device).float() for key, value in input_dict.items()}

    with torch.no_grad():
        correspondences = matcher(input_dict)

    mask = correspondences['confidence'] > 0.3
    indices = torch.nonzero(mask)
    correspondences['confidence'] = correspondences['confidence'][indices]
    correspondences['keypoints0'] = correspondences['keypoints0'][indices]
    correspondences['keypoints1'] = correspondences['keypoints1'][indices]
    correspondences['batch_indexes'] = correspondences['batch_indexes'][indices]

    mkpts0 = correspondences['keypoints0'].cpu().numpy()
    mkpts1 = correspondences['keypoints1'].cpu().numpy()
    H, inliers = cv2.findFundamentalMat(mkpts0, mkpts1, cv2.USAC_MAGSAC, 0.9, 0.99, 220000)
    inliers = inliers > 0
    del cut_img1, cut_img2, input_dict, correspondences, mask, indices
    return (mkpts0 / (new_w / old_size[0], new_h / old_size[1]) + (start_w, start_h),
            mkpts1 / (new_w / old_size[0], new_h / old_size[1]) + (start_w, start_h),
            inliers)


def loftr_matcher(path_img_1, path_img_2, matcher, w, h, n_pair, crp_w, crp_h, device, limit_printing=False):
    # n_pair
    # Load images
    img1 = load_torch_image(path_img_1, w, h)
    img2 = load_torch_image(path_img_2, w, h)


    img1 = img1.to(device)
    img2 = img2.to(device)

    if torch.cuda.is_available():
        matcher = matcher.to(device)

    crp_1_img = split_image(img1, n_pair)
    crp_2_img = split_image(img2, n_pair)

    # delete img1 & img2

    dict_keys = list(crp_1_img.keys())

    inliers, mkpts0, mkpts1 = [], [], []
    
    limit = int(n_pair/2)**2 if not limit_printing else limit_printing

    for pair_index in range(limit):
        pair_mkpts0, pair_mkpts1, pair_inliers = match_loftr_crop(
            crp_1_img[dict_keys[pair_index]]['image'],
            crp_2_img[dict_keys[pair_index]]['image'],
            crp_w,
            crp_h,
            matcher,
            [crp_1_img[dict_keys[0]]['image'].size()[3], crp_1_img[dict_keys[0]]['image'].size()[2]],
            crp_1_img[dict_keys[pair_index]]['coordinates']['start_w'],
            crp_1_img[dict_keys[pair_index]]['coordinates']['start_h'],
            device
        )
        inliers.append(pair_inliers)
        mkpts0.append(pair_mkpts0)
        mkpts1.append(pair_mkpts1)

        inliers = np.vstack(inliers)
        mkpts0 = np.vstack(mkpts0)
        mkpts1 = np.vstack(mkpts1)
    
        draw_LAF_matches(
            KF.laf_from_center_scale_ori(torch.from_numpy(mkpts0).view(1, -1, 2),
                                         torch.ones(mkpts0.shape[0]).view(1, -1, 1, 1),
                                         torch.ones(mkpts0.shape[0]).view(1, -1, 1)),
    
            KF.laf_from_center_scale_ori(torch.from_numpy(mkpts1).view(1, -1, 2),
                                         torch.ones(mkpts1.shape[0]).view(1, -1, 1, 1),
                                         torch.ones(mkpts1.shape[0]).view(1, -1, 1)),
            torch.arange(mkpts0.shape[0]).view(-1, 1).repeat(1, 2),
            K.tensor_to_image(crp_1_img[dict_keys[pair_index]]['image']),
            K.tensor_to_image(crp_2_img[dict_keys[pair_index]]['image']),
            inliers,
            draw_dict={'inlier_color': (0.2, 1, 0.2),
                       'tentative_color': None,
                       'feature_color': (0.2, 0.2, 1), 'vertical': False})
            
        plt.axis('off')
        plt.show()
        plt.clf()
        inliers, mkpts0, mkpts1 = [], [], []

def match_lightglue_crop(tensor_img1, tensor_img2, new_w, new_h, matcher, extractor, old_size, start_w, start_h, device):
    cut_img1 = resize_torch_crop(tensor_img1, new_w, new_h)
    cut_img2 = resize_torch_crop(tensor_img2, new_w, new_h)

    if torch.cuda.is_available():
        cut_img1 = cut_img1.cuda(device).float()
        cut_img2 = cut_img2.cuda(device).float()

    with torch.no_grad():
            feats0, feats1, matches01 = match_pair(
                extractor, matcher,
                K.color.rgb_to_grayscale(cut_img1).to(device),
                K.color.rgb_to_grayscale(cut_img2).to(device),
                device)
    
    kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]

    m_kpts0, m_kpts1 = kpts0[matches[..., 0]].cpu().numpy(), kpts1[matches[..., 1]].cpu().numpy()
    kpts0, kpts1 = kpts0.cpu().numpy(), kpts1.cpu().numpy()
    reposition = lambda x : x / (new_w / old_size[0], new_h / old_size[1]) + (start_w, start_h)
    # return (m_kpts0, m_kpts1, kpts0, kpts1)
    return (reposition(m_kpts0),
            reposition(m_kpts1),
            reposition(kpts0),
            reposition(kpts1))

def plot_matches(kpts0, kpts1, color=None, lw=1.5, ps=4, a=1.0, labels=None, axes=None):
    """Plot matches for a pair of existing images.
    Args:
        kpts0, kpts1: corresponding keypoints of size (N, 2).
        color: color of each match, string or RGB tuple. Random if not given.
        lw: width of the lines.
        ps: size of the end points (no endpoint if ps=0)
        indices: indices of the images to draw the matches on.
        a: alpha opacity of the match lines.
    """
    fig = plt.gcf()
    if axes is None:
        ax = fig.axes
        ax0, ax1 = ax[0], ax[1]
    else:
        ax0, ax1 = axes
    if isinstance(kpts0, torch.Tensor):
        kpts0 = kpts0.cpu().numpy()
    if isinstance(kpts1, torch.Tensor):
        kpts1 = kpts1.cpu().numpy()
    assert len(kpts0) == len(kpts1)
    if color is None:
        color = matplotlib.cm.hsv(np.random.rand(len(kpts0))).tolist()
    elif len(color) > 0 and not isinstance(color[0], (tuple, list)):
        color = [color] * len(kpts0)

    if lw > 0:
        for i in range(len(kpts0)):
            line = matplotlib.patches.ConnectionPatch(
                xyA=(kpts0[i, 0], kpts0[i, 1]),
                xyB=(kpts1[i, 0], kpts1[i, 1]),
                coordsA=ax0.transData,
                coordsB=ax1.transData,
                axesA=ax0,
                axesB=ax1,
                zorder=1,
                color=color[i],
                linewidth=lw,
                clip_on=True,
                alpha=a,
                label=None if labels is None else labels[i],
                picker=5.0,
            )
            line.set_annotation_clip(True)
            fig.add_artist(line)

    # freeze the axes to prevent the transform to change
    ax0.autoscale(enable=False)
    ax1.autoscale(enable=False)

    if ps > 0:
        ax0.scatter(kpts0[:, 0], kpts0[:, 1], c=color, s=ps)
        ax1.scatter(kpts1[:, 0], kpts1[:, 1], c=color, s=ps)


def visualize_matches(resized_img1,
                      resized_img2,
                      img1_matches,
                      img2_matches,
                      color="lime",
                      lw=0.1):
    plt.figure(figsize=(16, 16), dpi=250)

    plt.subplot(1, 2, 1)
    plt.imshow(resized_img1)
    plt.title('Image 1 with matches')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(resized_img2)
    plt.title('Image 2 with matches')
    plt.axis('off')

    plot_matches(img1_matches, img2_matches, color=color, lw=lw)


def save_npy(arrays, filenames, save_dir):
    """
    Save the match arrays to .npy files for later use.
    """
    os.makedirs(save_dir, exist_ok=True)
    for arr, filename in zip(arrays, filenames):
        np.save(os.path.join(save_dir, filename), arr)

def load_npy(filenames, save_dir):
    arrays = []
    for filename in filenames:
        arrays.append(np.load(os.path.join(save_dir, filename)))
    return arrays


def lightglue_matcher(
    path_img_1, path_img_2, matcher, extractor, w, h, n_pair, crp_w, crp_h, device,
    save_dir, limit_printing=False, visualize=False, plotting_size=2000
):
    img1 = load_torch_image(path_img_1, w, h).to(device)
    img2 = load_torch_image(path_img_2, w, h).to(device)

    crp_1_img = split_image(img1, n_pair)
    crp_2_img = split_image(img2, n_pair)

    dict_keys = list(crp_1_img.keys())
    results = []

    limit = int(n_pair / 2) ** 2 if not limit_printing else limit_printing

    # Collect all matches and crop data
    img_1_matches = []
    img_2_matches = []
    img_1_kpts = []
    img_2_kpts = []

    for pair_index in range(limit):
        crp1 = crp_1_img[dict_keys[pair_index]]["image"]
        crp2 = crp_2_img[dict_keys[pair_index]]["image"]

        pair_mkpts0, pair_mkpts1, kpts0, kpts1 = match_lightglue_crop(
            crp1,
            crp2,
            crp_w,
            crp_h,
            matcher,
            extractor,
            [crp1.size(-1), crp1.size(-2)],
            crp_1_img[dict_keys[pair_index]]["coordinates"]["start_w"],
            crp_1_img[dict_keys[pair_index]]["coordinates"]["start_h"],
            device,
        )

        del crp1, crp2
        
        img_1_matches.append(pair_mkpts0)
        img_2_matches.append(pair_mkpts1)
        
        img_1_kpts.append(kpts0)
        img_2_kpts.append(kpts1)

    del img1, img2
    img_1_matches = np.vstack(img_1_matches)
    img_2_matches = np.vstack(img_2_matches)

    img_1_kpts = np.vstack(img_1_kpts)
    img_2_kpts = np.vstack(img_2_kpts)
    
    save_npy([img_1_matches, img_2_matches],
             ["img1_matches.npy", "img2_matches.npy"],
             save_dir)
    save_npy([img_1_kpts, img_2_kpts],
             ["img_1_kpts.npy", "img_2_kpts.npy"],
             save_dir)
    

def visualize_kpts(
    img,
    kpts,
    color="red"
):

    plt.figure(figsize=(16, 16), dpi=250)
    plt.imshow(img)
    plt.title('Image with keypoints')
    plt.axis('off')
    ax = plt.gca()
    if isinstance(kpts, torch.Tensor):
        kpts = kpts.cpu().numpy()
    ax.scatter(kpts[:, 0], kpts[:, 1], c=color, s=4, linewidths=0, alpha=1)
    plt.show()
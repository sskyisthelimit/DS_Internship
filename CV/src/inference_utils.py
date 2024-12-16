import os
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
    
    return (m_kpts0 / (new_w / old_size[0], new_h / old_size[1]) + (start_w, start_h),
            m_kpts1 / (new_w / old_size[0], new_h / old_size[1]) + (start_w, start_h))


def split_image(image, n, save_crops=False, saving_path=None):
    _, c, h, w = image.size()
    if n % 2 != 0:
        raise ValueError("N should be even number")

    step_h = h // (n // 2)
    step_w = w // (n // 2)

    cropped_images = {}
    for i in range(n // 2):
        for j in range(n // 2):
            start_h = i * step_h
            end_h = (i + 1) * step_h
            start_w = j * step_w
            end_w = (j + 1) * step_w

            cropped = image.narrow(2, start_h, end_h - start_h).narrow(3, start_w, end_w - start_w)
            coordinates = {'start_h': start_h, 'end_h': end_h, 'start_w': start_w, 'end_w': end_w}
            key = f'cropped_{i}_{j}'
            cropped_images[key] = {'coordinates': coordinates, 'image': cropped}
            filename = "_" + str(start_h) + "_" + str(end_h) + "_" + str(start_w) + "_" + str(end_w) + ".png"
            filepath = os.path.join(saving_path, filename)
            save_image(cropped, filepath)

    return cropped_images

def save_image(image, path):
    image_pil = F.to_pil_image(image)
    image_pil.save(path)

def lightglue_matcher(path_img_1, path_img_2, matcher, extractor, w, h, n_pair, crp_w, crp_h, device):
    img1 = load_torch_image(path_img_1, w, h).to(device)
    img2 = load_torch_image(path_img_2, w, h).to(device)

    crp_1_img = split_image(img1, n_pair)
    crp_2_img = split_image(img2, n_pair)

    dict_keys = list(crp_1_img.keys())
    m_kpts0, m_kpts1 = [], []
    num_match_pairs = 0
    for pair_index in range(0, 6):
        crp1 = crp_1_img[dict_keys[pair_index]]['image']
        crp2 = crp_2_img[dict_keys[pair_index]]['image']

        pair_mkpts0, pair_mkpts1 = match_lightglue_crop(
            crp1, crp2, crp_w, crp_h, matcher, extractor,
            [crp1.size(-1), crp1.size(-2)],
            crp_1_img[dict_keys[pair_index]]['coordinates']['start_w'],
            crp_1_img[dict_keys[pair_index]]['coordinates']['start_h'],
            device
        )

        m_kpts0.append(pair_mkpts0)
        m_kpts1.append(pair_mkpts1)
        
        fig, axes = plt.subplots(
            1, 2, figsize=(10, 5), dpi=100, gridspec_kw={"width_ratios": [1, 1]}
        )
        
        # Plot the images
        axes[0].imshow(crp1.squeeze(0).T.cpu(), cmap="gray")
        axes[1].imshow(crp2.squeeze(0).T.cpu(), cmap="gray")
        
        # Clear axes ticks and frames
        for ax in axes:
            ax.axis("off")

        # Plot the matches
        viz2d.plot_matches(pair_mkpts0, pair_mkpts1, color="lime", lw=0.2, axes=axes)
        
        # Finalize and show
        plt.show()
        plt.close(fig)


def match_cutImg(tensor_img1, tensor_img2, new_w, new_h, matcher, old_size, start_w, start_h, device):
    cut_img1 = resize_torch_cutImage(tensor_img1, new_w, new_h)
    cut_img2 = resize_torch_cutImage(tensor_img2, new_w, new_h)

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


def matcher(path_img_1, path_img_2, matcher, w, h, n_pair, crp_w, crp_h, device):
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
    
    for pair_index in range(0, 6):
        pair_mkpts0, pair_mkpts1, pair_inliers = match_cutImg(
            crp_1_img[dict_keys[pair_index]]['image'],
            crp_2_img[dict_keys[pair_index]]['image'],
            crp_w,
            crp_h,
            matcher,
            [crp_1_img[dict_keys[0]]['image'].size()[3], crp_1_img[dict_keys[0]]['image'].size()[2]],
            crp_1_img[dict_keys[pair_index]]['coordinates']['start_w'],
            crp_1_img[dict_keys[pair_index]]['coordinates']['start_h']
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
import torch
import random
import numpy as np
import time

np.random.seed(33)
random.seed(33)
torch.manual_seed(33)

from torch.utils.data import Dataset, DataLoader

import json
from inference_utils import (load_torch_image,
                             split_image,
                             K)
# lightglue doesn't show up as a module in kaggle
# (i need manually cd into dir and import modules),
# so import line below would cause an error

# so to run evaluation you need define same initialize_models and import LightGlue, SuperPoint
# from inference import initialize_models

import collections.abc as collections
import glob
import itertools

class MatchingDataset(Dataset):
    def __init__(self, image_folder_path=None,
                 all_tiles_paths=None):
        super().__init__()
        if image_folder_path:
            all_tiles_paths = []
            for image_path in glob.iglob(image_folder_path + '**/*.jp2', recursive=True):
                all_tiles_paths.append(image_path)
            self.tile_pairs = list(itertools.combinations(all_tiles_paths, 2))
        elif isinstance(all_tiles_paths, list):
            self.tile_pairs = list(itertools.combinations(all_tiles_paths, 2))
        else:
            raise ValueError("Invalid init parameters")
    
    def __len__(self):
        return len(self.tile_pairs)
    
    def __getitem__(self, index):
        return self.tile_pairs[index]


def rbd(data: dict) -> dict:
    """Remove batch dimension from elements in data"""
    return {
        k: v[0] if isinstance(v, (torch.Tensor, np.ndarray, list)) else v
        for k, v in data.items()
    }


def map_tensor(input_, func):
    string_classes = (str, bytes)
    if isinstance(input_, string_classes):
        return input_
    elif isinstance(input_, collections.Mapping):
        return {k: map_tensor(sample, func) for k, sample in input_.items()}
    elif isinstance(input_, collections.Sequence):
        return [map_tensor(sample, func) for sample in input_]
    elif isinstance(input_, torch.Tensor):
        return func(input_)
    else:
        return input_


def batch_to_device(batch: dict, device: str = "cpu", non_blocking: bool = True):
    """Move batch (dict) to device"""

    def _func(tensor):
        return tensor.to(device=device, non_blocking=non_blocking).detach()

    return map_tensor(batch, _func)


def evaluate_metrics(extractor, matcher, img1, img2, device):
    """Evaluate key metrics for the SuperPoint + LightGlue pipeline."""
    start_time = time.time()

    feats0 = extractor.extract(img1)
    feats1 = extractor.extract(img2)

    extraction_time = time.time() - start_time

    start_matching = time.time()

    matches01 = matcher({"image0": feats0, "image1": feats1})
    data = [feats0, feats1, matches01]
    # remove batch dim and move to target device
    feats0, feats1, matches01 = [batch_to_device(rbd(x), device) for x in data]
    matching_time = time.time() - start_matching

    total_time = time.time() - start_time

    matching_scores = matches01["scores"]
    kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]

    matched_kpts0, matched_kpts1 = kpts0[matches[..., 0]].cpu().numpy(), kpts1[matches[..., 1]].cpu().numpy()
    kpts0, kpts1 = kpts0.cpu().numpy(), kpts1.cpu().numpy()
    # Match Coverage
    match_coverage_img1 = len(matched_kpts0) / len(kpts0) if len(kpts0) > 0 else 0
    match_coverage_img2 = len(matched_kpts1) / len(kpts1) if len(kpts1) > 0 else 0

    # Keypoint Displacement
    displacements = np.linalg.norm(matched_kpts0 - matched_kpts1, axis=1)
    avg_displacement = np.mean(displacements) if len(displacements) > 0 else 0
    max_displacement = np.max(displacements) if len(displacements) > 0 else 0

    # Mean Matching Score
    mean_matching_score = np.mean(matching_scores.cpu().numpy()) if len(matching_scores) > 0 else 0

    # Processing Time
    metrics = {
        "match_coverage_img1": match_coverage_img1,
        "match_coverage_img2": match_coverage_img2,
        "avg_displacement": avg_displacement,
        "max_displacement": max_displacement,
        "mean_matching_score": mean_matching_score,
        "extraction_time": extraction_time,
        "matching_time": matching_time,
        "total_time": total_time,
        "matches_len": matched_kpts0.shape[0],
        "kpts0_len": kpts0.shape[0],
        "kpts1_len": kpts1.shape[0],
    }

    return metrics

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def eval_on_jp2_tiles(args):
    g = torch.Generator()
    g.manual_seed(0)

    eval_dataset = MatchingDataset(args["image_folder_path"])
    eval_dl = DataLoader(
        eval_dataset,
        batch_size=1,
        num_workers=4,
        worker_init_fn=seed_worker,
        generator=g,
    )
    

    device = args["device"]
    extractor, matcher = initialize_models(device, args["max_num_keypoints"])

    # Macro metrics aggregation
    macro_metrics = {
        "match_coverage_img1": [],
        "match_coverage_img2": [],
        "avg_displacement": [],
        "max_displacement": [],
        "mean_matching_score": [],
        "extraction_time": [],
        "matching_time": [],
        "total_time": [],
        "matches_len": [],
        "kpts0_len": [],
        "kpts1_len": [],
    }
    
    for batch_idx, (path_img_1, path_img_2) in enumerate(eval_dl):
        print(f"Evaluating pair {batch_idx + 1}/{len(eval_dl)}: {path_img_1[0]} and {path_img_2[0]}")
    
        img1 = load_torch_image(path_img_1[0], args["width"], args["height"]).to(device)
        img2 = load_torch_image(path_img_2[0], args["width"], args["height"]).to(device)
    
        crp_1_img = split_image(img1, args["n_pair"])
        crp_2_img = split_image(img2, args["n_pair"])
    
        dict_keys = list(crp_1_img.keys())
    
        # Initialize accumulators for pair-level metrics
        pair_metrics = {
            "matches_len": 0,
            "kpts0_len": 0,
            "kpts1_len": 0,
            "match_coverage_img1": 0,
            "match_coverage_img2": 0,
            "avg_displacement_sum": 0,
            "max_displacement": 0,
            "mean_matching_score_sum": 0,
            "crop_count": 0,
        }
    
        start_time = time.time()  # Start timing for the full pair
        limit = int(args["n_pair"] / 2) ** 2

        for pair_index in range(limit):
            crp1 = crp_1_img[dict_keys[pair_index]]["image"]
            crp2 = crp_2_img[dict_keys[pair_index]]["image"]
    
            if torch.cuda.is_available():
                crp1 = crp1.cuda(device).float()
                crp2 = crp2.cuda(device).float()
    
            metrics = evaluate_metrics(
                extractor,
                matcher,
                K.color.rgb_to_grayscale(crp1).to(device),
                K.color.rgb_to_grayscale(crp2).to(device),
                device
            )
    
            # Accumulate metrics for the full pair
            pair_metrics["matches_len"] += metrics["matches_len"]
            pair_metrics["kpts0_len"] += metrics["kpts0_len"]
            pair_metrics["kpts1_len"] += metrics["kpts1_len"]
            pair_metrics["match_coverage_img1"] += metrics["match_coverage_img1"]
            pair_metrics["match_coverage_img2"] += metrics["match_coverage_img2"]
            pair_metrics["avg_displacement_sum"] += metrics["avg_displacement"] * metrics["matches_len"]
            pair_metrics["mean_matching_score_sum"] += metrics["mean_matching_score"] * metrics["matches_len"]
            pair_metrics["max_displacement"] = max(pair_metrics["max_displacement"], metrics["max_displacement"])
            pair_metrics["crop_count"] += 1
            del crp1, crp2
    
        total_time = time.time() - start_time  # Total time for the full pair
    
        # Finalize metrics for the pair
        match_coverage_img1 = pair_metrics["match_coverage_img1"] / pair_metrics["crop_count"]
        match_coverage_img2 = pair_metrics["match_coverage_img2"] / pair_metrics["crop_count"]
        avg_displacement = (
            pair_metrics["avg_displacement_sum"] / pair_metrics["matches_len"]
            if pair_metrics["matches_len"] > 0
            else 0
        )
        mean_matching_score = (
            pair_metrics["mean_matching_score_sum"] / pair_metrics["matches_len"]
            if pair_metrics["matches_len"] > 0
            else 0
        )
    
        # Add finalized pair-level metrics to macro_metrics
        macro_metrics["match_coverage_img1"].append(match_coverage_img1)
        macro_metrics["match_coverage_img2"].append(match_coverage_img2)
        macro_metrics["avg_displacement"].append(avg_displacement)
        macro_metrics["max_displacement"].append(pair_metrics["max_displacement"])
        macro_metrics["mean_matching_score"].append(mean_matching_score)
        macro_metrics["matches_len"].append(pair_metrics["matches_len"])
        macro_metrics["kpts0_len"].append(pair_metrics["kpts0_len"])
        macro_metrics["kpts1_len"].append(pair_metrics["kpts1_len"])
        macro_metrics["extraction_time"].append(metrics["extraction_time"])  # Keep using per-pair timing
        macro_metrics["matching_time"].append(metrics["matching_time"])
        macro_metrics["total_time"].append(total_time)

    del img1, img2
   
    final_macro_metrics = {key: np.mean(values) for key, values in macro_metrics.items()}

    # Save macro metrics as JSON
    save_path = f"{args['save_dir']}/macro_metrics.json"
    with open(save_path, "w") as f:
        json.dump(final_macro_metrics, f, indent=4)
    print(f"Macro metrics saved to {save_path}")

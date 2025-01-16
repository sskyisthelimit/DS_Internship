import torch
import random
import numpy as np

np.random.seed(33)
random.seed(33)
torch.manual_seed(33)

from utils import (MatchingDataset, DataLoader, evaluate_metrics)
import argparse

from inference import initialize_models
import json
from inference_utils import (load_torch_image,
                             split_image,
                             K)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def eval_on_jp2_tiles(args):
    g = torch.Generator()
    g.manual_seed(0)

    eval_dataset = MatchingDataset(args.image_folder_path)
    eval_dl = DataLoader(
        eval_dataset,
        batch_size=1,
        num_workers=4,
        worker_init_fn=seed_worker,
        generator=g,
    )

    device = args.device
    extractor, matcher = initialize_models(device, args.max_num_keypoints)

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

        img1 = load_torch_image(path_img_1[0], args.width, args.height).to(device)
        img2 = load_torch_image(path_img_2[0], args.width, args.height).to(device)

        crp_1_img = split_image(img1, args.n_pair)
        crp_2_img = split_image(img2, args.n_pair)

        dict_keys = list(crp_1_img.keys())

        for pair_index in range(args.n_pair):
            crp1 = crp_1_img[dict_keys[pair_index]]["image"]
            crp2 = crp_2_img[dict_keys[pair_index]]["image"]

            if torch.cuda.is_available():
                crp1 = crp1.cuda(device).float()
                crp2 = crp2.cuda(device).float()

            metrics = evaluate_metrics(
                extractor,
                matcher,
                K.color.rgb_to_grayscale(crp1).to(device),
                K.color.rgb_to_grayscale(crp2).to(device)
            )

            # Update macro metrics
            for key in macro_metrics.keys():
                macro_metrics[key].append(metrics[key])

            del crp1, crp2

        del img1, img2

    # Calculate averages for macro metrics
    final_macro_metrics = {key: np.mean(values) for key, values in macro_metrics.items()}

    # Save macro metrics as JSON
    save_path = f"{args.save_dir}/macro_metrics.json"
    with open(save_path, "w") as f:
        json.dump(final_macro_metrics, f, indent=4)
    print(f"Macro metrics saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Eval script for keypoint detection and matching.")
    parser.add_argument("--image_folder_path", type=str, required=True, help="Path to the image files dir.")
    parser.add_argument("--width", type=int, default=10980, help="Width of the image.")
    parser.add_argument("--height", type=int, default=10980, help="Height of the image.")
    parser.add_argument("--n_pair", type=int, default=20, help="Number of pairs.")
    parser.add_argument("--crop_width", type=int, default=1098, help="Crop width.")
    parser.add_argument("--crop_height", type=int, default=1098, help="Crop height.")
    parser.add_argument("--save_dir", type=str, default="./", help="Directory to save eval results.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use (e.g., 'cuda:0' or 'cpu').")
    parser.add_argument("--max_num_keypoints", type=int, default=1500, help="Maximal number of keypoints")
    
    args = parser.parse_args()
    eval_on_jp2_tiles(args)

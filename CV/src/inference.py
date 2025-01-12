import argparse
import cv2
from lightglue import LightGlue, SuperPoint
from inference_utils import (
    visualize_kpts, load_npy,
    lightglue_matcher, visualize_matches
)

def initialize_models(device):
    extractor_model = SuperPoint(max_num_keypoints=1500).eval().to(device)
    matcher_model = LightGlue(features="superpoint").eval().to(device)
    return extractor_model, matcher_model

def process_images(args):
    device = args.device
    extractor, matcher = initialize_models(device)

    matches_filenames = ["img1_matches.npy", "img2_matches.npy"]
    kpts_filenames = ["img_1_kpts.npy", "img_2_kpts.npy"]

    lightglue_matcher(
        path_img_1=args.filepath1,
        path_img_2=args.filepath2,
        matcher=matcher,
        extractor=extractor,
        w=args.width,
        h=args.height,
        n_pair=args.n_pair,
        crp_w=args.crop_width,
        crp_h=args.crop_height,
        device=device,
        save_dir=args.save_dir,
        limit_printing=None,
        matches_filenames=matches_filenames,
        kpts_filenames=kpts_filenames
    )

    img1 = cv2.cvtColor(cv2.imread(args.filepath1), cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(cv2.imread(args.filepath2), cv2.COLOR_BGR2RGB)

    plotting_size = args.out_img_size
    resized_img1 = cv2.resize(img1, (plotting_size, plotting_size), interpolation=cv2.INTER_AREA)
    resized_img2 = cv2.resize(img2, (plotting_size, plotting_size), interpolation=cv2.INTER_AREA)

    img_1_matches, img_2_matches = load_npy(
        filenames=matches_filenames,
        save_dir=args.save_dir
    )

    img_1_matches *= [plotting_size / img1.shape[1], plotting_size / img1.shape[0]]
    img_2_matches *= [plotting_size / img2.shape[1], plotting_size / img2.shape[0]]

    visualize_matches(resized_img1, resized_img2,
                      img_1_matches, img_2_matches,
                      save_path=args.match_img_path)
    
    kpts1, kpts2 = load_npy(filenames=kpts_filenames,
                            save_dir=args.save_dir)

    kpts1 *= [plotting_size / img1.shape[1], plotting_size / img1.shape[0]]
    kpts2 *= [plotting_size / img2.shape[1], plotting_size / img2.shape[0]]

    visualize_kpts(resized_img1,
                   kpts1,
                   save_path=args.kpts_img1_path)

    visualize_kpts(resized_img2,
                   kpts2,
                   color=(0, 0, 255),
                   save_path=args.kpts_img2_path)

    print("Keypoints and matches processed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference script for keypoint detection and matching.")
    parser.add_argument("--filepath1", type=str, required=True, help="Path to the first image file.")
    parser.add_argument("--filepath2", type=str, required=True, help="Path to the second image file.")
    parser.add_argument("--width", type=int, default=10980, help="Width of the image.")
    parser.add_argument("--height", type=int, default=10980, help="Height of the image.")
    parser.add_argument("--n_pair", type=int, default=20, help="Number of pairs.")
    parser.add_argument("--crop_width", type=int, default=1098, help="Crop width.")
    parser.add_argument("--crop_height", type=int, default=1098, help="Crop height.")
    parser.add_argument("--out_img_size", type=int, default=2000, help="Size for visualization and saving per image.")
    parser.add_argument("--save_dir", type=str, default="./", help="Directory to save .npy outputs.")
    parser.add_argument("--match_img_path", type=str, default="./matches.png", help="Path to save matches image.")
    parser.add_argument("--kpts_img1_path", type=str, default="./kpts1.png", help="Path to save kpts image 1.")
    parser.add_argument("--kpts_img2_path", type=str, default="./kpts2.png", help="Path to save kpts image 2.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use (e.g., 'cuda:0' or 'cpu').")
    
    args = parser.parse_args()
    process_images(args)

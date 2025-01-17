Ensure you have the following installed:

- Python 3.7 or later
- Step 0: create virtual env and activate it (i will work on docker image soon):
  ```
  python3 -m venv virt
  source virt/bin/activate
  ```
- Step 1 (clone repo):
  ```
  git clone https://github.com/sskyisthelimit/DS_Internship.git
  ```
- Step 2 (cd into CV task and install dependencies):
  ```
  cd DS_Internship/CV/
  pip install -r requirements.txt
  ```

- Step 3 (clone LightGlueTune and install in editable mode):
  ```
  !git clone --quiet git clone --quiet https://github.com/sskyisthelimit/LightGlueTune.git
  %cd LightGlueTune
  !pip install --progress-bar off --quiet -e .
  ```

- Step 4 (cd in src folder of DS_Internship/CV):
  ```
  cd src
  ```
- Step 5: run inference (everything explained below).

## Script Overview

The script performs the following steps:
1. Loads two input images.
2. Detects keypoints in each image using the SuperPoint model.
3. Matches the keypoints between the two images using the LightGlue model.
4. Visualizes and saves:
   - Keypoints for each image.
   - Matches between the images.

## Command-Line Arguments

| Argument             | Type   | Default       | Description                                               |
|----------------------|--------|---------------|-----------------------------------------------------------|
| `--filepath1`        | string | Required      | Path to the first image file.                             |
| `--filepath2`        | string | Required      | Path to the second image file.                            |
| `--width`            | int    | Required      | Width of the images.                                      |
| `--height`           | int    | Required      | Height of the images.                                     |
| `--do_fullsize`      | bool   | `False`       | Perform matching without image splitting if set to `True`.|
| `--n_pair`           | int    | `20`          | image will be splitted to (n_pair / 2) ** 2 crops.        |
| `--crop_width`       | int    | `1098`        | Crop width.                                               |
| `--crop_height`      | int    | `1098`        | Crop height.                                              |
| `--out_img_size`     | int    | `2000`        | Size for visualizations and saving per image.             |
| `--save_dir`         | string | `./`          | Directory to save `.npy` outputs.                         |
| `--match_img_path`   | string | `./matches.png` | Path to save the matches image.                         |
| `--kpts_img1_path`   | string | `./kpts1.png` | Path to save keypoints image for the first image.         |
| `--kpts_img2_path`   | string | `./kpts2.png` | Path to save keypoints image for the second image.        |
| `--device`           | string | `cuda:0`      | Device to use (`cuda:0` for GPU or `cpu`).                |
| `--max_num_keypoints`| int    | `1500`        | Maximum number of keypoints to detect per crop or full img|

## Usage

### Example Command

Run the script with the following command:

```bash
python3 -m inference \
  --filepath1 path/to/image1.jpg \
  --filepath2 path/to/image2.jpg \
  --width 10980 \
  --height 10980 \
  --save_dir ../outputs_example \
  --match_img_path ../outputs_example/matches.png \
  --kpts_img1_path ../outputs_example/kpts1.png \
  --kpts_img2_path ../outputs_example/kpts2.png
```

### Explanation

- `--filepath1` and `--filepath2` specify the paths to the images to process.
- `--save_dir` specifies the directory to save intermediate `.npy` outputs.
- `--match_img_path`, `--kpts_img1_path`, and `--kpts_img2_path` define the paths for saving visualization images.
- Additional arguments allow fine-tuning of processing settings (e.g., cropping dimensions, number of keypoints).

## Outputs

1. **Keypoints Files**:
   - Saved as `.npy` files in the specified `--save_dir`.
   - Filenames: `img_1_kpts.npy` and `img_2_kpts.npy`.

2. **Matches Files**:
   - Saved as `.npy` files in the specified `--save_dir`.
   - Filenames: `img1_matches.npy` and `img2_matches.npy`.

3. **Visualization Images**:
   - Keypoints images saved at paths defined by `--kpts_img1_path` and `--kpts_img2_path`.
   - Matches visualization saved at the path defined by `--match_img_path`.
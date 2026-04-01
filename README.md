# Image Stitching with SIFT + Deep Seam Composition

A UAV image stitching pipeline that combines SIFT-based feature matching with a deep learning composition network to produce seamless panoramic images.

## Overview

The pipeline works in two stages:

1. **Geometric alignment** — SIFT keypoint detection with FAISS-accelerated HNSW nearest-neighbor matching and RANSAC homography estimation. 

2. **Seam composition** — Siamese-Residual Mask Network (`SRMN.py`) predicts a mask for image stitching, producing a seamless composite without visible seams.

## Project Structure

```
.
├── many_stitching4_SIFT_Melted.py   # Main entry point: sequential multi-image stitching
├── functional_.py                   # Core utilities: SIFT matching, warping, blending,
│                                    # keyframe selection, homography optimization (Ceres)
├── SRMN.py                          # network definition + inference
├── requirements.txt                 # Python dependencies
└── model/
    └── epoch200_model.pth           # Pretrained model weights
```

## Requirements

Python 3.8+ recommended.

```bash
pip install -r requirements.txt
```

Key dependencies:
- `torch` >= 2.5
- `opencv-python` >= 4.10
- `faiss-cpu` >= 1.9
- `lightglue` (installed from GitHub, see `requirements.txt`)
- `pyceres` >= 2.4

## Usage

```bash
python many_stitching4_SIFT_Melted.py \
    --file_path /path/to/input/images/ \
    --log_path  /path/to/output/ \
    --weight_path_COMP ./model \
    --COMP True \
    --gpu 0
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--file_path` | *(required)* | Directory containing input images (`.jpg`, `.jpeg`, `.png`) |
| `--log_path` | *(required)* | Output directory for intermediate and final results |
| `--weight_path_COMP` | `./model` | Directory containing the composition model `.pth` |
| `--COMP` | `True` | Enable deep learning seam composition |
| `--cuda` | `False` | Use CUDA for inference |
| `--gpu` | `0` | GPU device index |
| `--ract_mask` | `False` | Use rectangular overlap mask for composition |

## How It Works

### Stage 1 — Feature Matching & Homography

For each image pair, SIFT features are extracted and matched using a FAISS HNSW index with Lowe's ratio test (threshold 0.7). A homography is estimated via RANSAC. 

### Stage 2 — Deep Seam Composition

The warped image pair and their binary masks are fed into the network SRMN. The network processes both images through shared encoder stages (RSU7 → RSU6 → RSU5 → RSU4 → RSU4F) and decodes the difference features to predict a stitching mask. The final composite is:

```
result = warp1 * learned_mask1 + warp2 * learned_mask2
```

Masks are upsampled to the original resolution via bicubic interpolation (or optionally LIIF implicit neural representation for higher quality).

An optional brightness normalization step (`light_ave=True`) equalizes the mean luminance of both images in the overlap region before composition.


"""
Visualization utilities for spine landmark data.
"""

import os
from typing import Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .landmarks import decode_landmarks_136, _to_pixel


def visualize_sample(
    idx: int,
    img_root: str = "./data/train",
    filenames_csv: str = "./data/train_txt/filenames.csv",
    landmarks_csv: str = "./data/train_txt/landmarks.csv",
    angles_csv: Optional[str] = "./data/train_txt/angles.csv",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (6, 10),
    show_points: bool = True,
    show_pairs: bool = True,
    show_quads: bool = True,
    show_centers: bool = True,
    label_vertebra: bool = True,
    point_size: int = 10,
    line_width: float = 1.2,
):
    """
    Visualize a single sample with vertebra landmarks and geometry.
    
    Args:
        idx: Sample index to visualize
        img_root: Root directory containing training images
        filenames_csv: Path to filenames CSV
        landmarks_csv: Path to landmarks CSV (136-dim per row)
        angles_csv: Path to angles CSV (optional)
        save_path: If provided, save figure to this path
        figsize: Figure size (width, height)
        show_points: Display all 68 landmark points
        show_pairs: Display left-right point connectors
        show_quads: Display vertebra quadrilaterals
        show_centers: Display vertebra center points
        label_vertebra: Label vertebrae with indices (0-16)
        point_size: Size of scatter points
        line_width: Width of lines
    
    Displays:
        - All 68 points (red x)
        - Left-right boundary connections (lines)
        - 17 vertebra quadrilaterals (blue)
        - Vertebra centers (cyan)
        - Vertebra indices 0-16 (lime text)
    """
    fn = pd.read_csv(filenames_csv, header=None)
    lm = pd.read_csv(landmarks_csv, header=None)
    ag = pd.read_csv(angles_csv, header=None) if angles_csv is not None else None

    name = str(fn.iloc[idx, 0])
    img_path = os.path.join(img_root, name)

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {img_path}")
    H, W = img.shape

    v136 = lm.iloc[idx].values.astype(np.float32)
    pts_norm, left_norm, right_norm, corners_norm = decode_landmarks_136(v136)

    pts = _to_pixel(pts_norm, W, H)
    left = _to_pixel(left_norm, W, H)
    right = _to_pixel(right_norm, W, H)

    corners = corners_norm.reshape(-1, 2)
    corners = _to_pixel(corners, W, H).reshape(17, 4, 2)  # (17, 4, 2) pixels

    # centers
    centers = corners.mean(axis=1)  # (17, 2)

    # title: filename + angles
    title = f"idx={idx} | {name}"
    if ag is not None:
        a0, a1, a2 = ag.iloc[idx].values.tolist()
        title += f"\nangles = [{a0:.2f}, {a1:.2f}, {a2:.2f}]"

    plt.figure(figsize=figsize)
    plt.imshow(img, cmap="gray")
    plt.title(title)
    plt.axis("off")

    # 1) all points
    if show_points:
        plt.scatter(pts[:, 0], pts[:, 1], s=point_size, c="red", marker="x", label="all 68 pts")

    # 2) pair connectors (left-right per layer)
    if show_pairs:
        for j in range(left.shape[0]):  # 34 layers
            plt.plot([left[j, 0], right[j, 0]], [left[j, 1], right[j, 1]], "-", linewidth=line_width)

    # 3) vertebra quads
    if show_quads:
        for i in range(17):
            quad = corners[i]  # LU, RU, LL, RL
            # draw as LU->RU->RL->LL->LU (closed quadrilateral)
            poly = np.stack([quad[0], quad[1], quad[3], quad[2], quad[0]], axis=0)
            plt.plot(poly[:, 0], poly[:, 1], "-", linewidth=line_width * 1.4)

    # 4) centers + labels
    if show_centers:
        plt.scatter(centers[:, 0], centers[:, 1], s=point_size * 2, c="cyan", marker="o", label="centers")

    if label_vertebra:
        for i in range(17):
            x, y = centers[i]
            plt.text(x, y, str(i), color="lime", fontsize=10, ha="center", va="center")

    plt.legend(loc="lower right")

    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    plt.show()

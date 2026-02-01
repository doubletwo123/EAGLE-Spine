"""
Geometric computation utilities for spine centerline analysis.
"""

import numpy as np
import torch
from typing import Tuple


def compute_centerline_geometry(
    corners: np.ndarray,
    device: str = "cpu"
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute centerline geometry from vertebra corners for SREG and LODA-Conv.
    
    Args:
        corners: (17, 4, 2) vertebra corners in pixel coordinates [LU, RU, LL, RL]
        device: torch device for output tensors
    
    Returns:
        Tuple containing:
            - centers: (1, 17, 2) vertebra center points [x, y]
            - tangents: (1, 17, 2) tangent vectors (normalized)
            - normals: (1, 17, 2) normal vectors (normalized)
            - ratios: (1, 17) aspect ratios (width / height)
    """
    # Compute centers
    centers = corners.mean(axis=1)  # (17, 2)
    
    # Compute tangent vectors (along the spine direction)
    tangents = np.zeros((17, 2), dtype=np.float32)
    for i in range(17):
        if i == 0:
            # First vertebra: use direction to next
            tangents[i] = centers[i + 1] - centers[i]
        elif i == 16:
            # Last vertebra: use direction from previous
            tangents[i] = centers[i] - centers[i - 1]
        else:
            # Middle vertebrae: average of incoming and outgoing directions
            tangents[i] = (centers[i + 1] - centers[i - 1]) / 2.0
    
    # Normalize tangent vectors
    tangent_norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    tangent_norms = np.maximum(tangent_norms, 1e-6)  # Avoid division by zero
    tangents = tangents / tangent_norms
    
    # Compute normal vectors (perpendicular to tangents)
    # Rotate tangent by 90 degrees: (x, y) -> (-y, x)
    normals = np.stack([-tangents[:, 1], tangents[:, 0]], axis=1)
    
    # Compute aspect ratios (width / height)
    ratios = np.zeros(17, dtype=np.float32)
    for i in range(17):
        quad = corners[i]  # (4, 2) [LU, RU, LL, RL]
        # Width: average of upper and lower widths
        upper_width = np.linalg.norm(quad[1] - quad[0])  # RU - LU
        lower_width = np.linalg.norm(quad[3] - quad[2])  # RL - LL
        avg_width = (upper_width + lower_width) / 2.0
        
        # Height: average of left and right heights
        left_height = np.linalg.norm(quad[2] - quad[0])  # LL - LU
        right_height = np.linalg.norm(quad[3] - quad[1])  # RL - RU
        avg_height = (left_height + right_height) / 2.0
        
        # Ratio with numerical stability
        ratios[i] = avg_width / max(avg_height, 1e-3)
    
    # Convert to torch tensors with batch dimension
    centers_t = torch.from_numpy(centers).unsqueeze(0).to(device)  # (1, 17, 2)
    tangents_t = torch.from_numpy(tangents).unsqueeze(0).to(device)  # (1, 17, 2)
    normals_t = torch.from_numpy(normals).unsqueeze(0).to(device)  # (1, 17, 2)
    ratios_t = torch.from_numpy(ratios).unsqueeze(0).to(device)  # (1, 17)
    
    return centers_t, tangents_t, normals_t, ratios_t


def normalize_image_size(img: np.ndarray, target_size: Tuple[int, int] = (512, 512)) -> Tuple[np.ndarray, float]:
    """
    Resize image to target size for network input.
    
    Args:
        img: Input image (H, W) or (H, W, C)
        target_size: Target (height, width)
    
    Returns:
        Tuple of:
            - Resized image
            - Scale factor (original / target)
    """
    import cv2
    
    H, W = img.shape[:2]
    target_h, target_w = target_size
    
    # Calculate scale factor
    scale = max(H / target_h, W / target_w)
    
    # Resize with aspect ratio preservation
    new_h = int(H / scale)
    new_w = int(W / scale)
    
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Pad to target size if needed
    if new_h < target_h or new_w < target_w:
        if len(img.shape) == 2:
            padded = np.zeros((target_h, target_w), dtype=img.dtype)
        else:
            padded = np.zeros((target_h, target_w, img.shape[2]), dtype=img.dtype)
        padded[:new_h, :new_w] = resized
        resized = padded
    
    return resized, scale

"""
Landmark coordinate processing utilities.
"""

from typing import Tuple
import numpy as np


def decode_landmarks_136(v136: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Decode 136-dimensional landmark vector into structured point representations.
    
    Args:
        v136: (136,) normalized coordinates with format:
            - First 68: x0..x67
            - Last 68: y0..y67
    
    Returns:
        Tuple containing:
            - pts: (68, 2) all normalized points [x, y]
            - left: (34, 2) left boundary points (odd indices)
            - right: (34, 2) right boundary points (even indices)
            - corners: (17, 4, 2) vertebra corners with order [LU, RU, LL, RL]
    
    Notes:
        Point order: top-to-bottom by y-value, 2 points per layer (left, right) alternating.
        34 layers = 17 vertebrae × (upper layer + lower layer)
    """
    v = v136.astype(np.float32).reshape(-1)
    assert v.shape[0] == 136, "Expected 136 values."

    x = v[:68] #  First 68 are x-coordinates
    y = v[68:] # Last 68 are y-coordinates
    pts = np.stack([x, y], axis=1)  # (68, 2) normalized

    left = pts[0::2]   # (34, 2)
    right = pts[1::2]  # (34, 2)

    corners = []
    for i in range(17):
        ju = 2 * i
        jd = 2 * i + 1
        LU = left[ju]
        RU = right[ju]
        LL = left[jd]
        RL = right[jd]
        corners.append(np.stack([LU, RU, LL, RL], axis=0))
    corners = np.stack(corners, axis=0)  # (17, 4, 2)

    return pts, left, right, corners


def _to_pixel(pts_norm: np.ndarray, W: int, H: int) -> np.ndarray:
    """
    Convert normalized coordinates [0, 1] to pixel coordinates.
    
    Args:
        pts_norm: (N, 2) normalized points
        W: image width in pixels
        H: image height in pixels
    
    Returns:
        (N, 2) pixel coordinates
    """
    pts = pts_norm.copy()
    pts[:, 0] *= (W - 1)
    pts[:, 1] *= (H - 1)
    return pts

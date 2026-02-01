"""
LODA-Conv test runner.
"""

import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.lda_conv import LODAConv2d, _normalize


def run_demo():
    B, C, H, W = 2, 64, 128, 128
    N = 17

    x = torch.randn(B, C, H, W)

    centers = torch.stack(
        [
            torch.stack([torch.linspace(20, 100, N), torch.linspace(10, 110, N)], dim=-1),
            torch.stack([torch.linspace(18, 98, N), torch.linspace(12, 112, N)], dim=-1),
        ],
        dim=0,
    )

    d = centers[:, 1:, :] - centers[:, :-1, :]
    u = _normalize(d, eps=1e-6)
    t = torch.zeros(B, N, 2)
    t[:, 1:-1, :] = _normalize(u[:, :-1, :] + u[:, 1:, :], eps=1e-6)
    t[:, 0, :] = u[:, 0, :]
    t[:, -1, :] = u[:, -1, :]
    n = torch.stack([-t[..., 1], t[..., 0]], dim=-1)

    ratio = torch.ones(B, N) * 1.2

    layer = LODAConv2d(64, 64, kernel_size=3, stride=1, padding=1, init_beta=1.0)
    out = layer(
        x,
        centers=centers,
        tangent=t,
        normal=n,
        ratio=ratio,
        stride_to_image=None,
        detach_geometry=True,
        compute_loss=True,
        return_stats=True,
    )

    print("y:", out.y.shape)
    print("offsets:", out.offsets.shape)
    print("loss_ortho:", out.loss_ortho)
    print("stats:", out.stats)


if __name__ == "__main__":
    run_demo()

"""
SREG / DRE-Gating (v2) test runner.
"""

import math
import os
import sys
from typing import Optional

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.sreg import SREGGating


def make_spine_centerline(
    N: int = 17,
    dy: float = 12.0,
    x_amp: float = 18.0,
    freq: float = 1.0,
    phase: float = 0.0,
    tilt: float = 0.0,
    noise_std: float = 0.7,
    kink_at: Optional[int] = None,
    kink_strength: float = 0.0,
    kink_width: float = 1.0,
    seed: int = 0,
):
    g = torch.Generator().manual_seed(seed)
    i = torch.arange(N, dtype=torch.float32)
    y = i * dy

    t = i / (N - 1)
    x_smooth = x_amp * torch.sin(2 * math.pi * freq * t + phase) + tilt * (t - 0.5) * (N * 0.8)

    x_kink = torch.zeros_like(x_smooth)
    if kink_at is not None and kink_strength != 0.0:
        k = float(kink_at)
        w = float(kink_width)
        x_kink = kink_strength * (
            torch.tanh((i - k) / w) - torch.tanh((i - (k + 1.0)) / w)
        )

    x_noise = torch.randn(N, generator=g) * noise_std
    x = x_smooth + x_kink + x_noise
    return torch.stack([x, y], dim=-1)


def make_batch_realistic(B: int = 3, N: int = 17):
    c0 = make_spine_centerline(
        N=N,
        dy=12.0,
        x_amp=5.0,
        freq=0.5,
        phase=0.3,
        tilt=2.0,
        noise_std=0.5,
        kink_at=None,
        seed=10,
    )
    c1 = make_spine_centerline(
        N=N,
        dy=12.0,
        x_amp=22.0,
        freq=0.7,
        phase=-0.4,
        tilt=4.0,
        noise_std=0.8,
        kink_at=None,
        seed=20,
    )
    c2 = make_spine_centerline(
        N=N,
        dy=12.0,
        x_amp=14.0,
        freq=0.6,
        phase=0.1,
        tilt=3.0,
        noise_std=0.6,
        kink_at=10,
        kink_strength=18.0,
        kink_width=0.7,
        seed=30,
    )

    c = torch.stack([c0, c1, c2], dim=0)
    mask = torch.ones((B, N), dtype=torch.float32)
    mask[1, 0] = 0.0
    mask[2, -2:] = 0.0
    return c, mask


def run_demo():
    c, mask = make_batch_realistic(B=3, N=17)
    sreg = SREGGating(lam_min=0.1, init_tau=2.0)
    out = sreg(c, mask=mask, return_loss=True, detach_geometry=True)

    print("tau =", float(sreg.tau))
    print("rho range per sample:")
    for b in range(c.shape[0]):
        r = out.rho[b, 1:-1]
        print(f"  sample {b}: min={r.min().item():.4f}, max={r.max().item():.4f}, mean={r.mean().item():.4f}")
    print("scale (MAD):", out.scale)
    print("L_cont:", out.loss_cont)

    b = 2
    print("\nSample 2 (pathology kink) rho[1:-1]:")
    print(out.rho[b, 1:-1].detach().cpu())
    print("Sample 2 gate[1:-1]:")
    print(out.gate[b, 1:-1].detach().cpu())

    out2 = sreg(c * 2.0, mask=mask, return_loss=False, detach_geometry=True)
    max_diff = (out2.rho - out.rho).abs().max().item()
    print("\nScale invariance check: max |rho(2x)-rho| =", max_diff)


if __name__ == "__main__":
    run_demo()

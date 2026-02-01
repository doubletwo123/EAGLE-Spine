# =========================================================
# SREG / DRE-Gating 模块（PyTorch）v2
# 采用方案二：混合尺度（MAD + gamma * median）
#
# 核心：
#   rho_i    = 1 - cos(theta_i)
#   med      = median(rho)
#   mad      = median(|rho - med|)
#   S        = mad + gamma * med + eps
#   Gate_i   = lam_min + (1-lam_min) * exp( - rho_i / (tau * S) )
#   L_cont   = mean_{valid} [ Gate_i * rho_i ]
#
# 输入：
#   c:    (B, N, 2)  中心点序列（按椎体顺序）
#   mask: (B, N)     可选，1 表示有效椎体，0 表示缺失/填充
#
# 输出：
#   rho:   (B, N)
#   gate:  (B, N)
#   scale: (B,)  混合尺度 S
#   loss_cont: scalar
# =========================================================

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


def _safe_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return torch.sqrt(torch.clamp((x * x).sum(dim=-1, keepdim=True), min=eps))


def _cosine_sim(u: torch.Tensor, v: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    num = (u * v).sum(dim=-1)
    den = (_safe_norm(u, eps=eps).squeeze(-1) * _safe_norm(v, eps=eps).squeeze(-1)).clamp(min=eps)
    return num / den


def _masked_median(x: torch.Tensor, mask: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """
    x:    (B, N)
    mask: (B, N)  {0,1}
    返回： (B,)  沿 dim 的 masked median（精确：按有效计数取中位数/双中位平均）
    """
    if x.ndim != 2 or mask.ndim != 2:
        raise ValueError("x and mask must be 2D tensors (B, N).")
    if dim != 1:
        raise ValueError("This helper assumes dim=1 for shape (B,N).")

    B, N = x.shape
    mask_bool = mask.to(dtype=torch.bool)

    # 将无效置为 +inf，排序后无效会落在末尾
    x2 = x.clone()
    x2[~mask_bool] = float("inf")
    x_sorted, _ = torch.sort(x2, dim=1)

    valid_counts = mask_bool.sum(dim=1).clamp(min=1)  # (B,)
    lo = (valid_counts - 1) // 2
    hi = valid_counts // 2

    idx_lo = lo.view(B, 1)
    idx_hi = hi.view(B, 1)

    med_lo = torch.gather(x_sorted, 1, idx_lo).squeeze(1)
    med_hi = torch.gather(x_sorted, 1, idx_hi).squeeze(1)
    return 0.5 * (med_lo + med_hi)


@dataclass
class SREGOutput:
    rho: torch.Tensor
    gate: torch.Tensor
    scale: torch.Tensor
    med: torch.Tensor
    mad: torch.Tensor
    loss_cont: Optional[torch.Tensor] = None


class SREGGating(nn.Module):
    """
    Scale-free Relative Energy Gating (SREG / DRE-Gating), v2 (Hybrid Scale)

    混合尺度：
      med = median(rho)
      mad = median(|rho - med|)
      S   = mad + gamma * med + eps

    参数：
      lam_min: 门控下界
      init_tau: 初始温度（可学习，softplus保证正数）
      gamma: 混合系数（可学习或固定）
      gamma_mode: "learnable" 或 "fixed"
    """

    def __init__(
        self,
        lam_min: float = 0.1,
        init_tau: float = 1.0,
        gamma: float = 1.0,
        gamma_mode: Literal["learnable", "fixed"] = "learnable",
        eps: float = 1e-6,
    ):
        super().__init__()
        if not (0.0 <= lam_min < 1.0):
            raise ValueError("lam_min must be in [0,1).")
        if gamma < 0:
            raise ValueError("gamma must be non-negative.")

        self.lam_min = float(lam_min)
        self.eps = float(eps)

        # tau > 0 via softplus
        inv_tau = torch.log(torch.exp(torch.tensor(float(init_tau))) - 1.0)
        self._tau_raw = nn.Parameter(inv_tau)

        self.gamma_mode = gamma_mode
        if gamma_mode == "learnable":
            # gamma >= 0 via softplus
            inv_g = torch.log(torch.exp(torch.tensor(float(gamma))) - 1.0)
            self._gamma_raw = nn.Parameter(inv_g)
        elif gamma_mode == "fixed":
            self.register_buffer("_gamma_fixed", torch.tensor(float(gamma)))
        else:
            raise ValueError("gamma_mode must be 'learnable' or 'fixed'.")

    @property
    def tau(self) -> torch.Tensor:
        return F.softplus(self._tau_raw) + self.eps

    @property
    def gamma(self) -> torch.Tensor:
        if self.gamma_mode == "learnable":
            return F.softplus(self._gamma_raw)  # no eps needed, allow 0
        return self._gamma_fixed

    def forward(
        self,
        c: torch.Tensor,                       # (B, N, 2)
        mask: Optional[torch.Tensor] = None,   # (B, N) in {0,1}
        return_loss: bool = True,
        detach_geometry: bool = True,
    ) -> SREGOutput:
        if c.ndim != 3 or c.size(-1) != 2:
            raise ValueError("c must have shape (B, N, 2).")
        B, N, _ = c.shape
        device = c.device
        dtype = c.dtype

        if mask is None:
            mask = torch.ones((B, N), device=device, dtype=torch.float32)
        else:
            mask = mask.to(device=device, dtype=torch.float32)
            if mask.shape != (B, N):
                raise ValueError("mask must have shape (B, N).")

        c_used = c.detach() if detach_geometry else c

        # u: (B, N-1, 2)
        d = c_used[:, 1:, :] - c_used[:, :-1, :]
        u = d / _safe_norm(d, eps=self.eps)

        # rho_mid: (B, N-2)
        cos_th = _cosine_sim(u[:, :-1, :], u[:, 1:, :], eps=self.eps)
        rho_mid = 1.0 - cos_th

        # pad to (B, N): endpoints rho=0
        rho = torch.zeros((B, N), device=device, dtype=dtype)
        rho[:, 1:-1] = rho_mid

        # valid middle mask
        mid_mask = torch.zeros((B, N), device=device, dtype=torch.float32)
        mid_mask[:, 1:-1] = 1.0
        valid_mask = mask * mid_mask  # (B,N)

        # -------- Hybrid scale: S = MAD + gamma * median + eps --------
        med = _masked_median(rho, valid_mask, dim=1)  # (B,)
        dev = torch.abs(rho - med[:, None])
        mad = _masked_median(dev, valid_mask, dim=1)  # (B,)

        scale = mad + self.gamma * med + self.eps  # (B,)
        scale = scale.clamp(min=self.eps)  # safety

        # gate
        denom = (self.tau * scale).clamp(min=self.eps)  # (B,)
        gate = self.lam_min + (1.0 - self.lam_min) * torch.exp(-rho / denom[:, None])

        # invalid points: gate=1, rho=0 already
        gate = gate * valid_mask + (1.0 - valid_mask) * 1.0

        loss_cont = None
        if return_loss:
            num = (gate * rho * valid_mask).sum()
            den = valid_mask.sum().clamp(min=1.0)
            loss_cont = num / den

        return SREGOutput(
            rho=rho,
            gate=gate,
            scale=scale,
            med=med,
            mad=mad,
            loss_cont=loss_cont,
        )


# # ===========================
# # 快速自测（可选）
# # ===========================
# if __name__ == "__main__":
#     B, N = 3, 17
#     # 简单造点：y 递增，x 带一点弯曲
#     i = torch.arange(N, dtype=torch.float32)
#     y = i * 12.0
#     c = []
#     for b in range(B):
#         x = 10.0 * torch.sin(2 * 3.14159 * (i / (N - 1)) + 0.2 * b) + 0.5 * torch.randn(N)
#         if b == 2:
#             # 加个突变
#             x[9:12] += torch.tensor([12.0, 18.0, 10.0])
#         c.append(torch.stack([x, y], dim=-1))
#     c = torch.stack(c, dim=0)

#     mask = torch.ones(B, N)
#     mask[1, 0] = 0
#     mask[2, -2:] = 0

#     sreg = SREGGatingV2(lam_min=0.1, init_tau=1.0, gamma=1.0, gamma_mode="learnable")
#     out = sreg(c, mask=mask, return_loss=True, detach_geometry=True)

#     print("tau:", float(sreg.tau))
#     print("gamma:", float(sreg.gamma))
#     print("scale:", out.scale)
#     print("med:", out.med)
#     print("mad:", out.mad)
#     print("L_cont:", out.loss_cont)


# # ===========================
# # 用法示例
# # ===========================
# if __name__ == "__main__":
#     B, N = 2, 17
#     c = torch.randn(B, N, 2)  # 例如由关键点均值得到的中心点序列
#     mask = torch.ones(B, N)
#     # 假设第2个样本缺失末端2个椎体
#     mask[1, -2:] = 0

#     sreg = SREGGating(lam_min=0.1, init_tau=1.0)
#     out = sreg(c, mask=mask, return_loss=True, detach_geometry=True)

#     print("tau =", float(sreg.tau))
#     print("rho shape:", out.rho.shape)
#     print("gate shape:", out.gate.shape)
#     print("scale:", out.scale)
#     print("L_cont:", out.loss_cont)

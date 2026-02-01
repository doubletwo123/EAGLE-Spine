from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torchvision.ops import deform_conv2d
except Exception as e:
    deform_conv2d = None


@dataclass
class LODAOutput:
    y: torch.Tensor                    # (B, C_out, H, W)
    offsets: torch.Tensor              # (B, 2*K, H, W)
    loss_ortho: Optional[torch.Tensor] # scalar
    stats: Optional[dict] = None       # 可选：调试用统计信息


def _normalize(v: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return v / (v.norm(dim=-1, keepdim=True).clamp_min(eps))


def _gather_offsets_at_centers(
    offsets: torch.Tensor,           # (B, 2K, H, W)
    centers_f: torch.Tensor,         # (B, N, 2) in feature coords (x,y)
) -> torch.Tensor:
    """
    在特征图坐标下的 centers 位置采样 offsets。
    为了简单稳健，我们用“最近邻取整 gather”（也可以改为 grid_sample 双线性）。
    返回: (B, N, K, 2)
    """
    B, twoK, H, W = offsets.shape
    assert twoK % 2 == 0
    K = twoK // 2
    N = centers_f.shape[1]

    # clamp & round to nearest pixel
    x = centers_f[..., 0].round().long().clamp(0, W - 1)  # (B,N)
    y = centers_f[..., 1].round().long().clamp(0, H - 1)  # (B,N)

    # gather: offsets[b, :, y, x] -> (B, 2K, N)
    # 先把 H,W 展平再索引更方便
    offsets_flat = offsets.view(B, twoK, H * W)  # (B,2K,HW)
    idx = (y * W + x).unsqueeze(1).expand(B, twoK, N)     # (B,2K,N)
    gathered = torch.gather(offsets_flat, dim=2, index=idx)  # (B,2K,N)

    # reshape to (B,N,K,2)
    gathered = gathered.permute(0, 2, 1).contiguous()  # (B,N,2K)
    gathered = gathered.view(B, N, K, 2)
    return gathered


class LODAConv2d(nn.Module):
    """
    LODA-Conv: Loop-free Orthogonal Deformable Alignment Convolution

    - offsets 由 g(F) 预测，不输入任何几何量（避免 forward 闭环）
    - 正交散布约束只在椎体中心点处计算（避免需要全图椎体归属）
    - 几何量默认 detach（避免梯度循环风险）

    约束：
      d_k = <Δp_k, n_i>,  s_k = <Δp_k, t_i>
      Dn_i = sqrt(Var_k(d_k)+eps), Dt_i = sqrt(Var_k(s_k)+eps)
      L_ortho = mean_i || Dn_i/(Dt_i+eps) - beta * r_i ||^2
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        eps: float = 1e-6,
        init_beta: float = 1.0,
    ):
        super().__init__()
        if deform_conv2d is None:
            raise RuntimeError(
                "torchvision.ops.deform_conv2d 不可用。请升级 torchvision 或安装支持的版本。"
            )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, int) else kernel_size[0]
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.eps = eps

        K = self.kernel_size * self.kernel_size
        self.K = K

        # offset predictor: (B, 2K, H, W)
        self.offset_conv = nn.Conv2d(
            in_channels, 2 * K,
            kernel_size=3, stride=stride, padding=1, bias=True
        )

        # conv weights for deform_conv2d (like nn.Conv2d parameters)
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, self.kernel_size, self.kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

        # beta: scale factor in ortho constraint (learnable scalar)
        self._beta_raw = nn.Parameter(torch.log(torch.exp(torch.tensor(init_beta)) - 1.0))

    @property
    def beta(self) -> torch.Tensor:
        # beta >= 0
        return F.softplus(self._beta_raw)

    def forward(
        self,
        x: torch.Tensor,                         # (B, C_in, H, W)
        centers: Optional[torch.Tensor] = None,   # (B, N, 2) in image coords or feature coords
        tangent: Optional[torch.Tensor] = None,   # (B, N, 2)
        normal: Optional[torch.Tensor] = None,    # (B, N, 2)
        ratio: Optional[torch.Tensor] = None,     # (B, N)
        stride_to_image: Optional[int] = None,    # 若 centers 给的是 image coords，这里传 feature stride
        detach_geometry: bool = True,
        compute_loss: bool = True,
        return_stats: bool = False,
    ) -> LODAOutput:
        B, Cin, H, W = x.shape
        offsets = self.offset_conv(x)  # (B, 2K, H', W') 注意：这里 stride=stride，形状会变
        # deform_conv2d 的输入输出空间跟 offsets 对齐，因此用 offsets 的 H,W
        H2, W2 = offsets.shape[-2], offsets.shape[-1]

        # 若 offset_conv stride != 1，x 也要同样下采样（或改 offset_conv stride=1）
        # 这里最稳妥：让 offset_conv stride = self.stride，与 deform_conv 的 stride 一致。
        y = deform_conv2d(
            input=x,
            offset=offsets,
            weight=self.weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            mask=None,
        )  # (B, C_out, H2, W2)

        loss_ortho = None
        stats = None

        if compute_loss:
            # 只有当几何量齐全时才算 loss
            if (centers is not None) and (tangent is not None) and (normal is not None) and (ratio is not None):
                # 把 centers 转到 feature coords（与 offsets 的空间一致）
                centers_f = centers
                if stride_to_image is not None:
                    # centers in image coords -> feature coords
                    centers_f = centers / float(stride_to_image)

                # 若 offsets 的空间因 stride 发生变化，需要进一步对齐
                # 例如：x 是 HxW，offsets 是 H2xW2（stride=2），则 feature coords 也应 / stride
                if self.stride != 1:
                    centers_f = centers_f / float(self.stride)

                # detach geometry to avoid loop gradients
                t = tangent
                n = normal
                r = ratio
                if detach_geometry:
                    centers_f = centers_f.detach()
                    t = t.detach()
                    n = n.detach()
                    r = r.detach()

                # normalize t,n
                t = _normalize(t, eps=self.eps)
                n = _normalize(n, eps=self.eps)

                # gather offsets at vertebra centers: (B,N,K,2)
                dP = _gather_offsets_at_centers(offsets, centers_f)

                # projections: d_k=<Δp_k,n>, s_k=<Δp_k,t>
                # t,n: (B,N,2) -> (B,N,1,2)
                t_exp = t.unsqueeze(2)
                n_exp = n.unsqueeze(2)

                d = (dP * n_exp).sum(dim=-1)  # (B,N,K)
                s = (dP * t_exp).sum(dim=-1)  # (B,N,K)

                # dispersions
                Dn = torch.sqrt(d.var(dim=-1, unbiased=False) + self.eps)  # (B,N)
                Dt = torch.sqrt(s.var(dim=-1, unbiased=False) + self.eps)  # (B,N)

                pred_ratio = Dn / (Dt + self.eps)  # (B,N)
                target_ratio = self.beta * r       # (B,N)

                loss_ortho = F.mse_loss(pred_ratio, target_ratio)

                if return_stats:
                    stats = {
                        "beta": float(self.beta.detach()),
                        "Dn_mean": float(Dn.mean().detach()),
                        "Dt_mean": float(Dt.mean().detach()),
                        "pred_ratio_mean": float(pred_ratio.mean().detach()),
                        "target_ratio_mean": float(target_ratio.mean().detach()),
                    }
            else:
                loss_ortho = None

        return LODAOutput(y=y, offsets=offsets, loss_ortho=loss_ortho, stats=stats)



"""Shared local-convolution / real-Mamba gated residual fusion for B2 and B3."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class LocalGlobalMambaFusion(nn.Module):
    """Fuse a shared local-convolution/Mamba projection with a gated residual.

    Input and output are BCHW; the Mamba branch scans the flattened spatial grid."""

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 64,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.0,
        residual_scale_init: float = 1e-2,
    ) -> None:
        super().__init__()
        if in_channels <= 0 or hidden_dim <= 0:
            raise ValueError("in_channels and hidden_dim must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if not isinstance(d_state, int) or not 1 <= d_state <= 256:
            raise ValueError("d_state must be an integer in [1, 256]")
        if d_conv not in (2, 3, 4):
            raise ValueError("d_conv must be 2, 3, or 4 for causal-conv1d")
        if not isinstance(expand, int) or expand < 1:
            raise ValueError("expand must be a positive integer")
        if not math.isfinite(residual_scale_init):
            raise ValueError("residual_scale_init must be finite")
        # Import only when B2/B3 is requested; B0/B1 and module inspection work
        # without installing CUDA extensions. Never substitute a fake SSM.
        try:
            from mamba_ssm import Mamba
        except (ImportError, OSError) as exc:
            raise ImportError(
                "B2/B3 require compatible Mamba CUDA extensions. "
                "Create the environment described in environment.yml and README.md."
            ) from exc
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim

        self.project_in = nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False)
        self.project_norm = nn.BatchNorm2d(hidden_dim)
        self.activation = nn.SiLU(inplace=False)

        # Local pathway: spatial 3x3 depthwise convolution + 1x1 mixing.
        self.local = nn.Sequential(
            nn.Conv2d(
                hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim, bias=False
            ),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=False),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1, bias=False),
        )

        # Global pathway: layer-normalized spatial sequence -> selective SSM.
        self.sequence_norm = nn.LayerNorm(hidden_dim)
        self.mamba = Mamba(d_model=hidden_dim, d_state=d_state, d_conv=d_conv, expand=expand)

        # Content-adaptive convex fusion: sigmoid gate has shape (B,d,H,W).
        self.gate = nn.Conv2d(2 * hidden_dim, hidden_dim, kernel_size=1)
        self.project_out = nn.Conv2d(hidden_dim, in_channels, kernel_size=1, bias=False)
        self.dropout = nn.Dropout2d(dropout)

        # Nonzero initial residual scale gives new layers gradients from step 1.
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale_init)))

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected BCHW input; got shape {tuple(x.shape)}")
        if not x.is_cuda:
            raise RuntimeError("B2/B3 require CUDA for the real Mamba selective-scan kernels")
        b, _, h, w = x.shape
        projected = self.activation(self.project_norm(self.project_in(x)))
        local = self.local(projected)

        # (B, d, H, W) -> (B, HW, d) -> Mamba -> (B, d, H, W)
        sequence = projected.flatten(2).transpose(1, 2).contiguous()
        sequence = self.sequence_norm(sequence)
        sequence = self.mamba(sequence)
        global_context = sequence.transpose(1, 2).reshape(b, -1, h, w)

        weights = torch.sigmoid(self.gate(torch.cat((local, global_context), dim=1)))
        mixed = weights * global_context + (1.0 - weights) * local
        residual = self.dropout(self.project_out(mixed))
        return x + self.residual_scale * residual

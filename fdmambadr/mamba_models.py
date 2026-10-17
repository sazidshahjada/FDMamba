"""Mamba-only CNN backbones for five-class diabetic-retinopathy grading.

This is an independent, classification-oriented implementation of local/global
feature fusion. It does NOT import MambaLiteUNet or require its source file.
The only state-space implementation used is the official ``mamba_ssm.Mamba``.

Supported timm architectures (randomly initialized; no pretrained weights):
    mobilenetv3     -> mobilenetv3_large_100
    efficientnet_b0 -> efficientnet_b0
    resnet50        -> resnet50

The module is inserted after ``backbone.forward_features`` and before the
original classifier head. All CNN convolutions remain unchanged: NO FDConv.

Install: pip install torch timm mamba-ssm causal-conv1d
Typical usage:
    model = MambaBackbone('resnet50', num_classes=5, hidden_dim=64)
    logits = model(torch.randn(2, 3, 224, 224).cuda())   # model on CUDA too

Note: CUDA/Triton versions required by mamba-ssm depend on the target machine.
"""
from __future__ import annotations

from typing import Dict
import math

import torch
from torch import Tensor, nn

from .backbone import Backbone


BACKBONES: Dict[str, str] = {
    "mobilenetv3": "mobilenetv3_large_100",
    "efficientnet_b0": "efficientnet_b0",
    "resnet50": "resnet50",
}


class LocalGlobalMambaFusion(nn.Module):
    """Lightweight residual fusion of local 2-D conv and global Mamba features.

    Args:
        in_channels: Channels in the backbone's final spatial feature tensor.
        hidden_dim: Bottleneck channels for each branch.
        d_state: Mamba state dimension.
        d_conv: Mamba's internal local convolution width.
        expand: Mamba expansion factor.
        dropout: Dropout on the fused residual branch.

    Input/output: (B, C, H, W) -> (B, C, H, W).
    Mamba sequences are raster-scanned over H*W positions; there is no attention
    or segmentation decoder. This is a new minimal design inspired by general
    local/global fusion, NOT a reproduction of MambaLiteUNet's AMF/LGFM/CGA.
    """

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
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1,
                      groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=False),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1, bias=False),
        )

        # Global pathway: layer-normalized spatial sequence -> selective SSM.
        self.sequence_norm = nn.LayerNorm(hidden_dim)
        self.mamba = Mamba(
            d_model=hidden_dim, d_state=d_state, d_conv=d_conv, expand=expand
        )

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


class MambaBackbone(nn.Module):
    """Scratch-trained CNN plus a single local/global Mamba fusion block.

    Call ``forward_features`` to obtain the fused spatial feature tensor and
    ``forward_head`` to classify it using the CNN's native head.
    """

    def __init__(
        self,
        name: str = "mobilenetv3",
        num_classes: int = 5,
        in_chans: int = 3,
        hidden_dim: int = 64,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.0,
        drop_rate: float = 0.0,
        residual_scale_init: float = 1e-2,
    ) -> None:
        super().__init__()
        base = Backbone(name, num_classes=num_classes, in_chans=in_chans,
                        drop_rate=drop_rate)
        self.name = base.name
        self.num_classes = num_classes
        self.backbone = base.model
        self.num_features = base.num_features
        self.fusion = LocalGlobalMambaFusion(
            in_channels=self.num_features, hidden_dim=hidden_dim,
            d_state=d_state, d_conv=d_conv, expand=expand, dropout=dropout,
            residual_scale_init=residual_scale_init,
        )

    def forward_features(self, x: Tensor) -> Tensor:
        features = self.backbone.forward_features(x)
        if features.ndim != 4:
            raise RuntimeError(
                f"Expected backbone.forward_features() to return BCHW, "
                f"but received shape {tuple(features.shape)}"
            )
        return self.fusion(features)

    def forward_head(self, features: Tensor) -> Tensor:
        return self.backbone.forward_head(features)

    def forward(self, x: Tensor) -> Tensor:
        return self.forward_head(self.forward_features(x))


def create_mamba_backbone(name: str = "mobilenetv3", **kwargs) -> MambaBackbone:
    """Convenience constructor with the same options as :class:`MambaBackbone`."""
    return MambaBackbone(name=name, **kwargs)


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise SystemExit("B2 smoke tests require a CUDA GPU and working Mamba kernels")
    device = torch.device("cuda")
    for architecture in BACKBONES:
        model = MambaBackbone(name=architecture, num_classes=5).to(device)
        model.eval()
        with torch.no_grad():
            prediction = model(torch.randn(2, 3, 224, 224, device=device))
        assert prediction.shape == (2, 5), (architecture, prediction.shape)
        total = sum(p.numel() for p in model.parameters())
        print(f"{architecture}: logits={tuple(prediction.shape)}, params={total:,}")

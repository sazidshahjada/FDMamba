"""B3: one backbone FDConv replacement plus the same Mamba fusion as B2.

The fusion block is placed after the backbone's forward_features and before the original classifier head.
All parameters are initialized from scratch; no segmentation modules are used.
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, Optional

import torch
from torch import Tensor, nn

from .fdconv_models import FDConvBackbone
from .mamba_models import LocalGlobalMambaFusion


class HybridBackbone(nn.Module):
    """Scratch-trained FDConv + Mamba classifier (B3).

    Uses precisely the B1 FDConv adapter and B2 local/global fusion block.
    A second FDConv in the fusion branch would confound the B2/B3 ablation.
    ``layer_path`` is relative to the underlying timm model; save the resolved
    ``config['backbone_fdconv']`` with every experiment.
    """

    def __init__(
        self,
        name: str = "mobilenetv3",
        num_classes: int = 5,
        in_chans: int = 3,
        drop_rate: float = 0.0,
        layer_path: Optional[str] = None,
        max_fdconv_channels: int = 384,
        fdconv_kernel_num: int = 4,
        fdconv_kwargs: Optional[Dict[str, Any]] = None,
        hidden_dim: int = 64,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        fusion_dropout: float = 0.0,
        residual_scale_init: float = 1e-2,
    ) -> None:
        super().__init__()
        self.backbone = FDConvBackbone(
            name=name, num_classes=num_classes, in_chans=in_chans,
            drop_rate=drop_rate, layer_path=layer_path,
            max_channels=max_fdconv_channels, kernel_num=fdconv_kernel_num,
            fdconv_kwargs=fdconv_kwargs,
        )
        self.name = self.backbone.name
        self.num_classes = num_classes
        self.num_features = self.backbone.num_features
        self.fusion = LocalGlobalMambaFusion(
            in_channels=self.num_features, hidden_dim=hidden_dim,
            d_state=d_state, d_conv=d_conv, expand=expand,
            dropout=fusion_dropout, residual_scale_init=residual_scale_init,
        )
        self.config = dict(
            name=self.name, num_classes=num_classes, in_chans=in_chans,
            drop_rate=drop_rate,
            backbone_fdconv=asdict(self.backbone.replacement),
            fdconv_kernel_num=fdconv_kernel_num,
            hidden_dim=hidden_dim, d_state=d_state, d_conv=d_conv,
            expand=expand, fusion_dropout=fusion_dropout,
            residual_scale_init=residual_scale_init,
            fusion_placement="after_forward_features_before_forward_head",
            fdconv_kwargs=dict(fdconv_kwargs or {}),
        )

    def forward_features(self, x: Tensor) -> Tensor:
        features = self.backbone.forward_features(x)
        if features.ndim != 4:
            raise RuntimeError(f"Backbone features must be BCHW, got {tuple(features.shape)}")
        return self.fusion(features)

    def forward_head(self, features: Tensor) -> Tensor:
        return self.backbone.forward_head(features)

    def forward(self, x: Tensor) -> Tensor:
        return self.forward_head(self.forward_features(x))


def create_hybrid_backbone(name: str = "mobilenetv3", **kwargs) -> HybridBackbone:
    """Construct the hybrid classifier from random initialization."""
    return HybridBackbone(name=name, **kwargs)


if __name__ == "__main__":
    # Run from repository root: python -m fdmambadr.hybrid_models
    if not torch.cuda.is_available():
        raise SystemExit("B3 smoke tests require a CUDA GPU and working Mamba kernels")
    for name in ("mobilenetv3", "efficientnet_b0", "resnet50"):
        model = HybridBackbone(name=name).cuda().eval()
        with torch.no_grad():
            output = model(torch.randn(1, 3, 224, 224, device="cuda"))
        assert output.shape == (1, 5), (name, tuple(output.shape))
        print(name, "shape:", tuple(output.shape), "parameters:", sum(p.numel() for p in model.parameters()))
        print("  CNN FDConv replacement:", model.config["backbone_fdconv"])

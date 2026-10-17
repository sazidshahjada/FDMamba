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

import torch

from .backbone import Backbone
from .fusion_backbone import FusionBackbone
from .mamba_fusion import LocalGlobalMambaFusion

BACKBONES: Dict[str, str] = {
    "mobilenetv3": "mobilenetv3_large_100",
    "efficientnet_b0": "efficientnet_b0",
    "resnet50": "resnet50",
}


class MambaBackbone(FusionBackbone):
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
        base = Backbone(name, num_classes=num_classes, in_chans=in_chans, drop_rate=drop_rate)
        self.name = base.name
        self.num_classes = num_classes
        self.backbone = base.model
        self.num_features = base.num_features
        self.fusion = LocalGlobalMambaFusion(
            in_channels=self.num_features,
            hidden_dim=hidden_dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout,
            residual_scale_init=residual_scale_init,
        )

        self.config = dict(
            name=name,
            num_classes=num_classes,
            in_chans=in_chans,
            hidden_dim=hidden_dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout,
            drop_rate=drop_rate,
            residual_scale_init=residual_scale_init,
        )


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

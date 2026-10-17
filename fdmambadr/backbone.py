"""CNN backbone factory for the FD-MambaDR project.

All networks are constructed with *randomly initialized weights*. ImageNet
pretraining or externally loaded checkpoints are intentionally not supported
by the factory. Checkpoints produced by your own experiments may be restored
using normal ``torch.load`` / ``load_state_dict`` workflows outside this file.

Requires:
    pip install torch timm

Model names:
    mobilenetv3  -> timm: mobilenetv3_large_100
    efficientnet  -> timm: efficientnet_b0
    resnet50      -> timm: resnet50

The default classification interface returns logits [B, num_classes]. The
feature interface returns a spatial tensor [B, C, H, W] using the same model
parameters, suitable for adding FDConv or a Mamba feature-fusion block.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import Tensor, nn

try:
    import timm
except ImportError as exc:
    raise ImportError("Backbone requires timm. Install it with: pip install timm") from exc


class Backbone(nn.Module):
    """Construct a randomly initialized CNN backbone for DR classification.

    Args:
        name: One of ``mobilenetv3``, ``efficientnet``, or ``resnet50``.
            Common aliases such as ``efficientnet_b0`` are accepted.
        num_classes: Number of DR classes (default 5).
        in_chans: Number of input image channels (default 3).
        drop_rate: Dropout probability used by the underlying classifier.

    Notes:
        * ``pretrained=False`` is hard-coded: weights are trained from scratch.
        * ``forward_features(x)`` returns the last spatial feature map.
        * ``forward_head(features)`` applies the original classifier head.
        * ``forward(x)`` returns logits by chaining those two operations.
        * Selective FDConv replacement should be done on ``self.model`` before
          training; the wrapper does not insert FDConv automatically.
    """

    _NAMES: Dict[str, str] = {
        "mobilenetv3": "mobilenetv3_large_100",
        "mobilenetv3_large": "mobilenetv3_large_100",
        "mobilenetv3_large_100": "mobilenetv3_large_100",
        "efficientnet": "efficientnet_b0",
        "efficientnet_b0": "efficientnet_b0",
        "resnet": "resnet50",
        "resnet50": "resnet50",
        "resnet_50": "resnet50",
    }

    def __init__(
        self,
        name: str,
        num_classes: int = 5,
        in_chans: int = 3,
        drop_rate: float = 0.0,
    ) -> None:
        super().__init__()
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")
        if in_chans <= 0:
            raise ValueError("in_chans must be positive")
        if not 0 <= drop_rate < 1:
            raise ValueError("drop_rate must be in [0, 1)")

        key = name.strip().lower().replace("-", "_")
        if key not in self._NAMES:
            supported = ", ".join(self.supported_names())
            raise ValueError(f"Unsupported backbone {name!r}. Supported: {supported}")

        self.config = dict(
            name=name, num_classes=num_classes, in_chans=in_chans, drop_rate=drop_rate
        )
        self.name = key
        self.timm_name = self._NAMES[key]
        self.num_classes = num_classes
        self.model = timm.create_model(
            self.timm_name,
            pretrained=False,  # Mandatory: all FD-MambaDR backbones train from scratch.
            num_classes=num_classes,
            in_chans=in_chans,
            drop_rate=drop_rate,
        )
        # MobileNetV3's num_features describes its classifier embedding
        # (1280), while forward_features stops before conv_head (960).
        self.num_features: int = int(
            self.model.conv_head.in_channels
            if self.timm_name == "mobilenetv3_large_100"
            else self.model.num_features
        )

        if not callable(getattr(self.model, "forward_features", None)):
            raise RuntimeError(f"{self.timm_name} lacks forward_features")
        if not callable(getattr(self.model, "forward_head", None)):
            raise RuntimeError(f"{self.timm_name} lacks forward_head")

    @classmethod
    def supported_names(cls) -> Tuple[str, ...]:
        """Return the canonical names for the supported backbone families."""
        return ("mobilenetv3", "efficientnet_b0", "resnet50")

    def forward_features(self, x: Tensor) -> Tensor:
        """Return the final spatial feature map, before the classifier head.

        Shapes: ``[B, 3, H, W] -> [B, C, h, w]``. The spatial dimensions
        depend on the backbone and input resolution.
        """
        return self.model.forward_features(x)

    def forward_head(self, features: Tensor) -> Tensor:
        """Classify a spatial feature map into DR severity logits."""
        return self.model.forward_head(features)

    def forward(self, x: Tensor) -> Tensor:
        """Return DR severity logits with shape ``[B, num_classes]``."""
        features = self.forward_features(x)
        return self.forward_head(features)


def create_backbone(
    name: str,
    num_classes: int = 5,
    in_chans: int = 3,
    drop_rate: float = 0.0,
) -> Backbone:
    """Convenience factory, equivalent to ``Backbone(...)``."""
    return Backbone(name, num_classes=num_classes, in_chans=in_chans, drop_rate=drop_rate)


if __name__ == "__main__":
    # Smoke test: python -m fdmambadr.backbone
    for backbone_name in Backbone.supported_names():
        net = create_backbone(backbone_name)
        net.eval()
        with torch.inference_mode():
            x = torch.randn(1, 3, 224, 224)
            feature_map = net.forward_features(x)
            logits = net.forward_head(feature_map)
        print(
            f"{backbone_name:18s} | features={tuple(feature_map.shape)} "
            f"| logits={tuple(logits.shape)} "
            f"| params={sum(p.numel() for p in net.parameters()):,}"
        )
        assert logits.shape == (1, 5)

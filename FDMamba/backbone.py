"""Scratch-initialized CNNs with native timm feature and classifier paths."""

from typing import Dict, Tuple

import timm
from torch import Tensor, nn


class Backbone(nn.Module):
    """MobileNetV3, EfficientNet-B0 or ResNet-50 returning raw class logits."""

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
            pretrained=False,  # Mandatory: all FDMamba backbones train from scratch.
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
        """Classify a spatial feature map into class logits."""
        return self.model.forward_head(features)

    def forward(self, x: Tensor) -> Tensor:
        """Return class logits with shape ``[B, num_classes]``."""
        features = self.forward_features(x)
        return self.forward_head(features)

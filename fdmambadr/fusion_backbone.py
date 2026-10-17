"""Common forward interface for the two fusion-enabled classifiers."""

from torch import Tensor, nn


class FusionBackbone(nn.Module):
    """Compose a feature backbone, shape-preserving fusion and native head.

    Subclasses construct ``backbone`` and ``fusion``. Keeping them in their
    original locations preserves parameter names in existing state dictionaries.
    """

    def forward_features(self, x: Tensor) -> Tensor:
        """Return the fused BCHW feature map before native pooling/projection."""
        features = self.backbone.forward_features(x)
        if features.ndim != 4:
            raise RuntimeError(f"Expected BCHW backbone features, got {tuple(features.shape)}")
        return self.fusion(features)

    def forward_head(self, features: Tensor) -> Tensor:
        """Apply the original backbone head and return raw logits."""
        return self.backbone.forward_head(features)

    def forward(self, x: Tensor) -> Tensor:
        """Classify a batch without applying softmax."""
        return self.forward_head(self.forward_features(x))

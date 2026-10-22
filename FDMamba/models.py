"""The four architecture variants and one public model factory."""

from copy import deepcopy
from dataclasses import asdict
from typing import Any, Dict, Optional

from torch import Tensor, nn

from .backbone import Backbone
from .fdconv import FDConvBackbone
from .fusion import LocalGlobalMambaFusion


class FusionBackbone(nn.Module):
    """Apply spatial fusion between the native backbone features and classifier."""

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


class MambaBackbone(FusionBackbone):
    """B2: scratch CNN with one local/global Mamba residual fusion block."""

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


class HybridBackbone(FusionBackbone):
    """B3: the B1 FDConv backbone followed by the same fusion block as B2."""

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
            name=name,
            num_classes=num_classes,
            in_chans=in_chans,
            drop_rate=drop_rate,
            layer_path=layer_path,
            max_channels=max_fdconv_channels,
            kernel_num=fdconv_kernel_num,
            fdconv_kwargs=fdconv_kwargs,
        )
        self.name = self.backbone.name
        self.num_classes = num_classes
        self.num_features = self.backbone.num_features
        self.fusion = LocalGlobalMambaFusion(
            in_channels=self.num_features,
            hidden_dim=hidden_dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=fusion_dropout,
            residual_scale_init=residual_scale_init,
        )
        self.config = dict(
            name=self.name,
            num_classes=num_classes,
            in_chans=in_chans,
            drop_rate=drop_rate,
            backbone_fdconv=asdict(self.backbone.replacement),
            fdconv_kernel_num=fdconv_kernel_num,
            hidden_dim=hidden_dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            fusion_dropout=fusion_dropout,
            residual_scale_init=residual_scale_init,
            fusion_placement="after_forward_features_before_forward_head",
            fdconv_kwargs=dict(fdconv_kwargs or {}),
        )


MODEL_CLASSES = {"B0": Backbone, "B1": FDConvBackbone, "B2": MambaBackbone, "B3": HybridBackbone}
BACKBONES = ("mobilenetv3", "efficientnet_b0", "resnet50")


def model_name(backbone: str, variant: str = "B3") -> str:
    """Return the public family name and ablation suffix."""
    canonical = Backbone._NAMES[backbone.strip().lower().replace("-", "_")]
    family = {
        "mobilenetv3_large_100": "MobileNetV3",
        "efficientnet_b0": "EfficientNetB0",
        "resnet50": "ResNet50",
    }[canonical]
    return f"FDMamba-{family}-{variant.upper()}"


def build_model(
    backbone="mobilenetv3",
    variant="B0",
    num_classes=5,
    *,
    pretrained=False,
    fdconv=None,
    mamba=None,
    **backbone_kwargs,
):
    """Build a native-head classifier and retain its reconstructible specification.

    ``fdconv`` uses B1 options: layer_path, max_channels, kernel_num,
    fdconv_kwargs. ``mamba`` uses B2 options: hidden_dim, d_state, d_conv,
    expand, dropout, residual_scale_init. B3 receives those same options.
    Inactive-variant options fail rather than being silently ignored.
    """
    if pretrained is not False:
        raise ValueError("Primary experiments require pretrained=False")
    variant = variant.upper()
    if variant not in MODEL_CLASSES:
        raise ValueError(f"Unknown variant {variant!r}; choose B0, B1, B2 or B3")
    fdconv, mamba = deepcopy(fdconv or {}), deepcopy(mamba or {})
    if fdconv and variant not in ("B1", "B3"):
        raise ValueError("FDConv options require B1 or B3")
    if mamba and variant not in ("B2", "B3"):
        raise ValueError("Mamba options require B2 or B3")
    allowed_fd = {"layer_path", "max_channels", "kernel_num", "fdconv_kwargs"}
    allowed_mamba = {"hidden_dim", "d_state", "d_conv", "expand", "dropout", "residual_scale_init"}
    if set(fdconv) - allowed_fd or set(mamba) - allowed_mamba:
        raise ValueError("Unknown FDConv/Mamba option; see build_model's docstring")
    if set(backbone_kwargs) - {"in_chans", "drop_rate"}:
        raise ValueError(
            "Backbone options are in_chans and drop_rate; use fdconv/mamba for variant options"
        )
    options = dict(backbone_kwargs)
    if variant == "B3":
        rename = {"max_channels": "max_fdconv_channels", "kernel_num": "fdconv_kernel_num"}
        options.update({rename.get(key, key): value for key, value in fdconv.items()})
        options.update(
            {"fusion_dropout" if key == "dropout" else key: value for key, value in mamba.items()}
        )
    else:
        options.update(fdconv)
        options.update(mamba)
    model = MODEL_CLASSES[variant](backbone, num_classes=num_classes, **options)
    if variant in ("B1", "B3"):
        adapted = model if variant == "B1" else model.backbone
        fdconv["layer_path"] = adapted.replacement.path
    model.build_config = dict(
        backbone=backbone,
        variant=variant,
        num_classes=num_classes,
        pretrained=False,
        fdconv=fdconv,
        mamba=mamba,
        **backbone_kwargs,
    )
    model.display_name = model_name(backbone, variant)
    return model

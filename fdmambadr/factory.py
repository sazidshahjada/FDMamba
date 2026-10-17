"""One scratch-only constructor for the B0/B1/B2/B3 experiment matrix."""

from __future__ import annotations

from copy import deepcopy

from .backbone import Backbone
from .fdconv_models import FDConvBackbone
from .hybrid_models import HybridBackbone
from .mamba_models import MambaBackbone

MODEL_CLASSES = {
    "B0": Backbone,
    "B1": FDConvBackbone,
    "B2": MambaBackbone,
    "B3": HybridBackbone,
}


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
    return model

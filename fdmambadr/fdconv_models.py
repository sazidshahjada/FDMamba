"""Selective FDConv-only CNN variants, with random initialization.

This replaces ONE compatible nn.Conv2d in an otherwise unchanged timm
backbone. It intentionally does not add any Mamba components.

The supplied Fourier/kernel modulation implementation is in fdmambadr/FDConv.py.
Standalone classification does not require MMCV.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn

from .backbone import Backbone


@dataclass(frozen=True)
class ReplacementInfo:
    path: str
    in_channels: int
    out_channels: int
    kernel_size: tuple[int, int]
    stride: tuple[int, int]
    padding: tuple[int, int]
    dilation: tuple[int, int]
    groups: int
    bias: bool
    original_parameters: int
    replacement_parameters: int


def _get_parent(root: nn.Module, path: str):
    parts = path.split('.')
    parent = root.get_submodule('.'.join(parts[:-1])) if len(parts) > 1 else root
    return parent, parts[-1]


def _compatible(path: str, module: nn.Module, *, max_channels: int,
                require_kernel: Optional[int] = None) -> bool:
    if not isinstance(module, nn.Conv2d):
        return False
    # The original FDConv reshapes grouped weights as though groups==1.
    if module.groups != 1 or module.in_channels <= 16 or module.out_channels <= 16:
        return False
    if max(module.in_channels, module.out_channels) > max_channels:
        return False
    if module.kernel_size not in ((1, 1), (3, 3)):
        return False
    if require_kernel is not None and module.kernel_size[0] != require_kernel:
        return False
    if (module.stride != (1, 1) or module.dilation != (1, 1)
            or module.padding_mode != 'zeros'):
        return False
    # Avoid classifier and stem locations, where features are at high resolution.
    if any(k in path.lower() for k in ('classifier', 'head', 'stem', 'conv_stem', 'conv1')):
        return False
    # SE convolutions operate on globally pooled 1x1 descriptors, not spatial
    # feature maps. Replacing one would change the intended FDConv ablation.
    if '.se.' in path.lower() or 'squeeze' in path.lower():
        return False
    return True


class FP32FDConv(nn.Module):
    """Compute frequency-domain operations in FP32, including under CUDA AMP.

    The supplied implementation uses FFT and creates float32 intermediates.
    This wrapper keeps the layer's external dtype contract, though the full
    backward and AMP performance require CUDA verification.
    """
    def __init__(self, layer: nn.Module):
        super().__init__()
        self.layer = layer

    def forward(self, x: Tensor) -> Tensor:
        from contextlib import nullcontext
        target_dtype = x.dtype
        device = x.device.type
        context = torch.autocast(device_type=device, enabled=False) if device in ('cuda', 'cpu') else nullcontext()
        with context:
            out = self.layer(x.float())
        return out.to(dtype=target_dtype)


class FDConvBackbone(Backbone):
    """CNN with a selected standard Conv2d replaced by true FDConv.

    Args:
        name: mobilenetv3, efficientnet_b0, resnet50.
        layer_path: Explicit path relative to ``self.model``, recommended
            for reproducible comparisons. When absent, automatically selects
            a compatible candidate and records ``replacement``.
        max_channels: Restricts expensive dynamic kernel construction.
        kernel_num: Number of FDConv frequency kernel groups.
        fdconv_kwargs: Extra supported FDConv constructor options.

    WARNING: Auto-selection is architecture/version dependent. Record the
    selected layer and pin timm version in every experiment configuration.
    """
    def __init__(self, name: str, num_classes: int = 5, in_chans: int = 3,
                 drop_rate: float = 0.0, layer_path: Optional[str] = None,
                 max_channels: int = 384, kernel_num: int = 4,
                 fdconv_kwargs: Optional[dict] = None):
        super().__init__(name, num_classes, in_chans, drop_rate)
        if not isinstance(kernel_num, int) or kernel_num < 1:
            raise ValueError('kernel_num must be a positive integer')
        try:
            from .FDConv import FDConv
        except ImportError as exc:
            raise ImportError('Cannot load fdmambadr.FDConv.') from exc
        candidates = [(path, layer) for path, layer in self.model.named_modules()
                      if _compatible(path, layer, max_channels=max_channels)]
        if layer_path is None:
            if self.timm_name == 'resnet50':
                preferred = [(p, m) for p, m in candidates if p.startswith('layer3.') and p.endswith('.conv2')]
            else:
                # Prefer 1x1 pointwise layers on depthwise-heavy architectures.
                preferred = [(p, m) for p, m in candidates if m.kernel_size == (1, 1)]
            choices = preferred or candidates
            if not choices:
                raise RuntimeError(f'No compatible FDConv site in {self.timm_name}; supply verified layer_path.')
            layer_path, original = choices[len(choices) // 2]
        else:
            try:
                original = self.model.get_submodule(layer_path)
            except AttributeError as exc:
                raise ValueError(f'No module named {layer_path!r} in {self.timm_name}') from exc
            if not _compatible(layer_path, original, max_channels=max_channels):
                raise ValueError(f'{layer_path!r} is incompatible with this FDConv adapter: {original}')
        # Exact Fourier construction partitions the real-FFT spectrum into
        # kernel_num groups; the count must be divisible by kernel_num.
        bins = original.out_channels * original.kernel_size[0] * (
            original.in_channels * original.kernel_size[1] // 2 + 1)
        if bins % kernel_num:
            raise ValueError(f'FFT coefficient count {bins} must be divisible by kernel_num={kernel_num}')
        cfg = dict(fdconv_kwargs or {})
        for reserved in ('in_channels', 'out_channels', 'kernel_size', 'stride', 'padding',
                         'dilation', 'groups', 'bias', 'kernel_num', 'convert_param'):
            if reserved in cfg:
                raise ValueError(f'{reserved} is derived from the existing convolution and cannot be overridden')
        new_layer = FDConv(
            in_channels=original.in_channels, out_channels=original.out_channels,
            kernel_size=original.kernel_size, stride=original.stride,
            padding=original.padding, dilation=original.dilation,
            groups=original.groups, bias=original.bias is not None,
            kernel_num=kernel_num, convert_param=True, **cfg,
        )
        if not new_layer.fdconv_active or not hasattr(new_layer, 'dft_weight'):
            raise ValueError(f'FDConv configuration disables frequency modeling at {layer_path!r}')
        self.replacement = ReplacementInfo(
            path=layer_path, in_channels=original.in_channels,
            out_channels=original.out_channels, kernel_size=original.kernel_size,
            stride=original.stride, padding=original.padding,
            dilation=original.dilation, groups=original.groups,
            bias=original.bias is not None,
            original_parameters=sum(p.numel() for p in original.parameters()),
            replacement_parameters=sum(p.numel() for p in new_layer.parameters()),
        )
        parent, key = _get_parent(self.model, layer_path)
        setattr(parent, key, FP32FDConv(new_layer))


def create_fdconv_backbone(name: str, **kwargs) -> FDConvBackbone:
    return FDConvBackbone(name, **kwargs)

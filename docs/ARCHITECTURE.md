# FDMamba architecture

`FDMamba.build_model(backbone, variant, num_classes)` is the single constructor.
It uses timm 0.9.16 without pretrained weights. The backbone's feature extraction,
pooling, projection and classifier stay intact. Grayscale images are replicated
to RGB in preprocessing; only the native classifier output width changes by dataset.

| Variant | Class | Computation |
|---|---|---|
| B0 | Backbone | Native CNN |
| B1 | FDConvBackbone | Native CNN with one eligible convolution replaced |
| B2 | MambaBackbone | CNN → local/global fusion → native head |
| B3 | HybridBackbone | B1 backbone → identical B2 fusion → native head |

All four classes and `build_model` are exported by `FDMamba`. Implementations are
organized into `backbone.py`, `fdconv.py`, `fusion.py` and `models.py`. Module nesting
and parameter names match the previous architecture. `FDConv.py` retains the
supplied standalone algorithm without numerical changes.

## FDConv

| Backbone | Replacement path | Channels | Kernel | Default FBM |
|---|---|---|---|---|
| MobileNetV3-Large | blocks.2.2.conv_pwl | 120 → 40 | 1×1 | Off |
| EfficientNet-B0 | blocks.2.0.conv_pwl | 144 → 40 | 1×1 | Off |
| ResNet-50 | layer3.3.conv2 | 256 → 256 | 3×3 | On |

FDW and KSM are active at each site. MobileNet/EfficientNet use ungrouped pointwise
convolutions because their spatial convolutions are depthwise. This difference
must be acknowledged in comparisons. The adapter preserves convolution semantics
and rejects unsupported/inactive configurations. FFT operations run in FP32 under
autocast. The factory records the resolved site in every saved model configuration.

The supplied FDConv includes previously existing standalone fixes for singleton
pooled descriptors, actual FFT grid sizes, odd inverse-FFT shapes, kernel grouping,
initialization and checkpointing. Those implementations were not changed by this rewrite.

## Mamba fusion

A shared 1×1 projection compresses the final spatial features to 64 channels by
default. One branch applies depthwise/pointwise local convolutions; the other
flattens the grid into a raster sequence, applies LayerNorm and real Mamba, then
reshapes it back. A sigmoid gate combines the two branches. A final projection,
optional dropout and learned residual scale (initially 0.01) produce the output.
The original backbone classifier consumes the fused features.

There is no segmentation decoder, bidirectional scan or additional FDConv inside
fusion. B2/B3 share precisely this block. Their real kernels require CUDA.
MobileNet's spatial feature width is 960, while its classifier embedding is 1280;
the original head preserves that projection.

Optional advanced settings can be specified in the training YAML:

```yaml
model:
  backbone: mobilenetv3
  variant: B3
  drop_rate: 0.2
  fdconv:
    layer_path: blocks.2.2.conv_pwl
    kernel_num: 4
  mamba:
    hidden_dim: 64
    d_state: 16
    d_conv: 4
    expand: 2
```

Inactive options are rejected: FDConv settings require B1/B3, Mamba settings require
B2/B3. Keep settings matched within the intended ablation. Refer to `build_model`
for accepted options and [verification](IMPLEMENTATION_STATUS.md) for tested limits.

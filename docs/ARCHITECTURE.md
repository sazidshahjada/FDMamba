# Architecture and extension guide

## Fixed model behavior

B0 is the exact selected timm family with a five-logit classifier and no pretrained
weights. `Backbone.forward_features` exposes spatial features; `forward_head`
retains native pooling, projection and classifier behavior. MobileNet's spatial
width is 960 while its native classifier embedding is 1280; these are intentionally
different and the optional dense wrapper preserves that projection.

B1 uses the supplied FDConv algorithm at one compatible ungrouped convolution.
The adapter preserves stride, padding, dilation, groups and bias semantics and
rejects silently inactive FDConv configurations. Fourier conversion starts from
a new random kernel. It does not transplant pretrained weights.

B2 and B3 share `LocalGlobalMambaFusion`:

1. Project final spatial CNN features to a configurable bottleneck (default 64).
2. Share that projection between a depthwise/pointwise local branch and a raster
   sequence of length `height * width` processed by LayerNorm and real Mamba.
3. Predict a sigmoid gate from the concatenated branch outputs.
4. Mix `gate * global + (1 - gate) * local`, project to the original channels,
   and add a learned scaled residual (initial scale 0.01).
5. Apply the original backbone head.

Fusion runs after `forward_features`, before `forward_head`. B3 adds only the
backbone FDConv relative to B2; it has no extra FDConv inside fusion. This preserves
the uploaded shared-projection adaptation, including its one-direction raster
scan. It is not an exact MambaLiteUNet reproduction. No segmentation code is used.

## Default FDConv sites (timm 0.9.16)

| Backbone | Path | Channels | Kernel | Default FBM |
|---|---|---|---|---|
| MobileNetV3-Large | `blocks.2.2.conv_pwl` | 120 to 40 | 1 x 1 | Off |
| EfficientNet-B0 | `blocks.2.0.conv_pwl` | 144 to 40 | 1 x 1 | Off |
| ResNet-50 | `layer3.3.conv2` | 256 to 256 | 3 x 3 | On |

FDW and KSM remain active at all three sites. Pointwise replacement in MobileNet
and EfficientNet is an existing compatibility choice: their spatial convolutions
are depthwise. This cross-backbone difference must be reported. Configurations
built through `build_model` record the resolved path so evaluation does not need
to repeat auto-selection. Pin timm and pass explicit sites for final experiments.

## Reuse without changing the experiment

- **New backbone:** extend `Backbone`'s name map and confirm spatial channel count,
  native head behavior and FDConv eligibility. Add forward/backward tests before
  using it in the experimental matrix.
- **New data source:** supply a deterministic `Dataset` and `DataLoader` to
  `Trainer.evaluate`. Preserve class order and saved normalization. Folder-based
  sources can use `create_eval_loader`; CSV-based sources need an adapter. Useful
  metadata attributes are `samples`, `targets`, `classes`, `class_to_idx`, `root`,
  `normalization_mean`, and `normalization_std`.
- **New fusion:** implement a BCHW-to-BCHW module, then deliberately register a new
  variant. Keep B2/B3 matched for the existing ablation.
- **New head:** use `BackboneWrapper`, with identical settings for paired variants.
- **New loss/optimizer/scheduler:** pass them to `Trainer`. Advanced callbacks and
  hooks remain available for notebook compatibility; ordinary runs need none.

`EpochRunner` is an internal execution base, not a second trainer API. The public
`Trainer` owns the model, loss, optimizer, configuration and checkpoint lifecycle.
Epoch losses use summed numerators and the appropriate denominator (CE target
weights or nonignored sample count). Gradient accumulation preserves that same
normalization, including the final partial accumulation window.

## Compatibility

Existing model class paths, factory convenience functions, `Trainer`, loss/metric
APIs and `utils.loaders.visualize_images` remain available. Moved fusion classes
are re-exported from their original modules. Model parameter paths and numerical
architecture are retained. New metadata makes some legacy **full trainer**
checkpoint signatures incompatible; exact old-run resume is not promised. The
original constructor plus a strict `model.load_state_dict` can restore compatible
model-only states for evaluation. Never strip checkpoint guards to force a resume.

The CLI intentionally covers native-head experiments. It does not dynamically
import arbitrary classes from YAML, duplicate twelve training pipelines, or fit
external statistics. A configured test loader is evaluated once after training
using the checkpoint selected on validation.

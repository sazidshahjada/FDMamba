# FD-MambaDR
FD-MambaDR: Frequency-Dynamic Convolution and State-Space Feature Modeling for Efficient Diabetic Retinopathy Grading

Read [project_overview.md](project_overview.md) for the research protocol and
[docs/IMPLEMENTATION_STATUS.md](docs/IMPLEMENTATION_STATUS.md) for the code review,
verified behavior, and remaining work. All classifiers use random initialization.

## Environment

The full GPU environment targets **Linux x86_64** (including WSL2), **Python 3.11**,
**PyTorch 2.5.1 / torchvision 0.20.1**, and **CUDA 12.4**. It requires an NVIDIA
GPU and a compatible installed driver. PyTorch's version pair follows its
[official installation matrix](https://pytorch.org/get-started/previous-versions/#v251).

```bash
conda env create -f environment.yml
conda activate fd-mambadr
python -m pip check
python -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; import selective_scan_cuda, causal_conv1d_cuda; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name())"
python -m unittest discover -s tests -v
```

The YAML includes data/image processing, metrics, plots, TensorBoard, and system
monitoring dependencies. It uses official prebuilt
[Mamba 2.2.4](https://github.com/state-spaces/mamba/releases/tag/v2.2.4) and
[causal-conv1d 1.5.4](https://github.com/Dao-AILab/causal-conv1d/releases/tag/v1.5.4)
wheels matched to CPython 3.11, PyTorch 2.5, CUDA 12, and the Conda PyTorch C++ ABI.
These are platform-specific binaries; changing Python/PyTorch/platform requires
matching replacement wheels or source builds using the installed PyTorch with
`--no-build-isolation`. A local CUDA compiler is unnecessary for these wheels.
MMCV is unnecessary for this standalone FDConv implementation.

On a CPU machine, B0/B1 can run with compatible `torch`, `torchvision`, and `timm`.
The test suite explicitly skips real Mamba execution when CUDA is unavailable.
B2/B3 require the actual Mamba CUDA kernels; they have no substitute implementation.

## Models

| Variant | Constructor | Components |
|---|---|---|
| B0 | `fdmambadr.backbone.Backbone` | CNN |
| B1 | `fdmambadr.fdconv_models.FDConvBackbone` | CNN with one FDConv replacement |
| B2 | `fdmambadr.mamba_models.MambaBackbone` | CNN and local/global Mamba fusion |
| B3 | `fdmambadr.hybrid_models.HybridBackbone` | B1 and the same fusion block as B2 |

Each supports `mobilenetv3`, `efficientnet_b0`, and `resnet50` and produces five
raw logits by default. Shared aliases are resolved by the baseline factory.

```python
import torch
from fdmambadr.hybrid_models import HybridBackbone

model = HybridBackbone("mobilenetv3", num_classes=5, hidden_dim=64).cuda()
logits = model(torch.randn(1, 3, 224, 224, device="cuda"))
print(logits.shape)  # torch.Size([1, 5])
print(model.config)  # Save this with the experiment/checkpoint.
```

To customize the dense classifier, pass a **backbone class** to `BackboneWrapper`:

```python
from torch import nn
from fdmambadr.fdconv_models import FDConvBackbone
from fdmambadr.wrapper import BackboneWrapper

model = BackboneWrapper(
    FDConvBackbone,
    name="mobilenetv3",
    num_classes=5,
    hidden_dims=(512, 128),
    dropout=(0.3, 0.1),
    activation=nn.GELU,
    normalization="layernorm",
    backbone_kwargs={"kernel_num": 4},
)
```

The same interface accepts `Backbone`, `MambaBackbone`, and `HybridBackbone`.
`hidden_dims=()` selects a single output linear layer; dropout can be one scalar
or a probability per hidden layer. Activation options go in `activation_kwargs`;
normalization accepts `None`, `"layernorm"`, or `"batchnorm"`. Dense BatchNorm
requires batches larger than one during training. Other backbone options, such
as Mamba's `hidden_dim`, go in `backbone_kwargs`.

The wrapper removes the original output classifier and keeps the native pooling
and projection, including MobileNet's 960-to-1280 channel projection. It returns
raw logits and records dense-head settings in `model.config`. Save these along
with the underlying backbone's resolved FDConv metadata when applicable, and
use identical dense heads across compared variants.

Keep model parameters in FP32 and use `torch.autocast`/`torch.amp.GradScaler` for
mixed precision. The FDConv wrapper runs its FFT and dynamic convolution in FP32.
The tests include real CUDA forward/backward, AMP, and optimizer checks for B2/B3
when a GPU is present; CPU tests cannot verify those paths.

## ImageFolder data loading

Supply the existing split directories; each contains class folders `0` to `4`.

```text
train/0/*.jpg   train/1/*.jpg   ... train/4/*.jpg
val/0/*.jpg     val/1/*.jpg     ... val/4/*.jpg
test/0/*.jpg    test/1/*.jpg    ... test/4/*.jpg
```

```python
from utils.loaders import create_dataloaders, visualize_images

loaders = create_dataloaders(
    train_path="/path/to/train",
    val_path="/path/to/val",
    test_path="/path/to/test",
    image_size=224,
    batch_size=32,
    num_workers=4,  # Use 0 for a simple notebook/debugging setup.
    augment=False,  # The current combined dataset already has offline augmentation.
)
images, labels = next(iter(loaders["train"]))
print(loaders["train"].dataset.class_to_idx)
print(loaders["train"].dataset.class_counts)
figure, axes = visualize_images(loaders, rows=3, cols=4, split="train")
```

Only training is shuffled. `augment=True` adds mild training-only flips, rotation
and color changes. Validation/test transforms are deterministic and retain every
sample, including the last incomplete batch. Missing split classes retain their
original labels; unexpected class folders fail. For other folder names, pass
`class_names` in the verified severity order.

Images are decoded as RGB float32 tensors. Defaults use pixel values in `[0, 1]`.
For normalization, fit once with
`mean, std = compute_train_mean_std(train_path, image_size=224)` (import from
`utils.loaders`), then pass `mean=mean, std=std` to `create_dataloaders` and save
them with the run. This reads the entire training folder once; it never fits
validation/test statistics. The grid function reverses normalization for display
and can save a figure with `save_path="outputs/grid.png", show=False`.

## Loss functions

All three losses in `utils/losses.py` accept raw logits `[batch, classes]` and
`torch.long` labels `[batch]`. They support `reduction="none"`, `"mean"`, or
`"sum"`, plus `ignore_index`. Mean reduction returns differentiable zero for an
empty/all-ignored batch. Half/bfloat16 logits use FP32 loss arithmetic.

```python
from utils.losses import CrossEntropy, BalancedCrossEntropy, FocalLoss

ce = CrossEntropy(label_smoothing=0.0)
balanced_ce = BalancedCrossEntropy.from_dataset(loaders["train"].dataset)
focal = FocalLoss(gamma=2.0, alpha=balanced_ce.weight)

loss = balanced_ce(logits, labels)
loss.backward()
```

`BalancedCrossEntropy` uses fixed weights `N / (C * class_count)` from the
**training split only**. You can also supply a count vector ordered by label
index, e.g. `BalancedCrossEntropy([55162, 18470, 24198, 7936, 9475])`. Zero-count
training classes fail explicitly. Counts/weights are buffers and are checkpointed
in `state_dict`; use the same weight policy across compared models.

`CrossEntropy` supports optional `weight` and `label_smoothing`, following
[PyTorch's cross-entropy definition](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html).
Its weighted mean divides by the sum of target-class weights. The multiclass
`FocalLoss` uses `-alpha_t * (1 - p_t)**gamma * log(p_t)`, adapting the
[focal-loss definition](https://arxiv.org/abs/1708.02002) to softmax classification.
`alpha` can be omitted, a uniform scalar, or a weight vector. Focal mean averages
over nonignored samples; alpha weights affect its numerator. With `gamma=0` and
no alpha, it equals plain cross-entropy. Record the chosen loss, reduction,
weights/counts, smoothing and focal settings in each run configuration.

## Classification metrics

`utils/metrics.py` provides a complete-split helper and an epoch accumulator:

```python
from utils.metrics import ClassificationMetrics, compute_metrics

# One call when predictions for the entire split are available:
report = compute_metrics(logits, labels)

# Or accumulate one update per batch and compute once at the end of the split:
meter = ClassificationMetrics(num_classes=5)
meter.update(logits, labels)  # Repeat for each batch.
report = meter.compute()
print(report["qwk"], report["macro_f1"], report["balanced_accuracy"])
meter.reset()  # Before the next epoch/split.
```

Inputs can be PyTorch tensors (including CUDA outputs) or NumPy arrays/lists.
Predictions are the **first** argument, matching the loss API. By default `[N,C]`
inputs are logits; use `from_logits=False` for probabilities. `[N]` integer
predictions are also supported. Targets are integer labels in the configured
class order; `ignore_index=-100` is excluded by default. GPU inputs are detached
and stored on CPU. Do not average QWK/F1 from separate minibatch reports.

The report contains accuracy, balanced accuracy, macro precision/recall/F1,
quadratic weighted kappa (`qwk`), per-grade precision/recall/F1/support, and the
fixed-size confusion matrix (true labels in rows, predictions in columns).
Macro scores include all configured classes with undefined per-class ratios set
to zero; balanced accuracy averages recall over represented true classes.
These conventions follow [scikit-learn's metrics](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics).

`referable_dr` reports sensitivity, specificity, ROC-AUC, support and a binary
confusion matrix for true grade `>=2`. With probability/logit inputs, it uses the
sum of grades 2–4 probabilities and a fixed threshold of 0.5. With hard-grade
inputs, the decision is predicted grade `>=2` and AUC is unavailable. Configure
the grade boundary/threshold before evaluating; use `referable_grade=None` to
disable this exploratory endpoint. No thresholds are fitted automatically.

Undefined QWK, AUC or binary rates return `None`, with an explanation in
`undefined_metrics`. An empty/all-ignored split raises an error. Handle undefined
QWK explicitly before selecting a checkpoint. Reports can be serialized with
`json.dump(report, file, allow_nan=False)`. For prediction exports,
`meter.get_predictions()` returns copies of target/predicted labels and class
probabilities as NumPy arrays in update order.

The repository implements models, ImageFolder loading/visualization, losses,
classification metrics, and regression tests. Manifest generation, training/resume,
QWK checkpoint selection, and evaluation/export scripts remain to be implemented.
The environment supplies their dependencies; it does not constitute a completed
training/evaluation pipeline.

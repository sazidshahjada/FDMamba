# FD-MambaDR

Scratch-trained, five-class diabetic retinopathy classification with three CNN
backbones and four controlled model variants. Start with
[project_overview.md](project_overview.md) for the research protocol and
[the review](docs/IMPLEMENTATION_STATUS.md) for verified behavior and limitations.

## Setup

For the supplied Linux/NVIDIA environment:

```bash
conda env create -f environment.yml
conda activate fdmambadr
python -m pip check
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -m unittest discover -s tests -v
```

The environment retains the supplied Python 3.11 / PyTorch 2.5.1 / CUDA 12.4
configuration and platform-specific Mamba wheels. Verify it on your GPU machine;
this revision ran on CPU. For B0/B1-only work, install a matching PyTorch 2.5.1 /
torchvision 0.20.1 wheel pair for your device, then `pip install -r requirements.txt`.
B2/B3 require real `mamba_ssm` and `causal_conv1d` CUDA extensions; there is no
surrogate fallback. MMCV is not required by the supplied FDConv file.

## Train and evaluate

Keep the publisher's existing splits. Each split directory contains verified
class folders `0`, `1`, `2`, `3`, `4`, with images inside. Missing classes in
validation/test keep their original label indices. Weighted training requires
all five classes to be present.

1. Edit paths, model variant, batch size and run directory in `configs/train.yaml`.
2. Train, select the checkpoint using validation QWK, and evaluate the test set:

```bash
python train.py --config configs/train.yaml
python train.py --config configs/train.yaml --resume
```

Training and evaluation print the selected device (including the GPU name on CUDA),
full model name, parameter counts, image size, class order, and each available
split's sample/batch counts, per-class counts, paths, batch size and worker count.
Each epoch shows training/validation progress bars with running loss, followed by
validation accuracy, macro-F1 and QWK. If `data.test_path` is configured, training
ends with test evaluation using `best.pt`, printing test loss, accuracy and macro-F1.

3. Optionally evaluate the fixed checkpoint separately or on external data:

```bash
python evaluate.py --checkpoint experiments/MNV3_B0_seed42/best.pt --data data/test --split test
python evaluate.py --checkpoint experiments/MNV3_B0_seed42/best.pt --data data/idrid --split idrid
```

Use a separate run directory for every backbone/variant/seed. `--resume` continues
`last.pt` in that directory; move/copy the entire directory when relocating a
run. Keep the original total-epoch schedule on resume. The test split is evaluated
only after the training loop. External class folders must reflect a verified
five-grade mapping.

The default recipe uses training-label inverse-frequency cross-entropy, AdamW,
warm-up/cosine scheduling, and validation-QWK early stopping. Exact QWK ties keep
the earlier checkpoint. Undefined QWK is recorded explicitly and cannot create
`best.pt`; automatic test evaluation uses `last.pt` if no validation optimum was
defined. Focal loss and unweighted CE remain available for deliberate ablations.

Images are RGB in `[0,1]` unless training-fitted mean/std are supplied. Normalization
is never fitted during evaluation. Additional online augmentation defaults to
**off in the supplied config**, because the current collection includes offline
augmentation. The existing loader API retains its original `augment=True` default.
Keep normalization, resolution, loss and head settings matched across comparisons.

Runs save `config.yaml`, `metadata.json`, `dataset_fingerprints.json`,
`history.csv/json`, `best.pt`, `last.pt`, and evaluation metrics/prediction CSVs.
When training finishes, including early stopping, both the validation-selected best
checkpoint (if defined) and the final checkpoint are also exported to
`training.saved_weights_dir` (default `saved_weights`). Filenames include the full
backbone, variant/method, run directory name, seed, epoch and `best`/`last`, for example:

```text
saved_weights/mobilenetv3_large_100_B0_Baseline_MNV3_B0_seed42_seed42_epoch0007_best.pt
saved_weights/mobilenetv3_large_100_B0_Baseline_MNV3_B0_seed42_seed42_epoch0022_last.pt
```

The files retain weights, class names, model construction and preprocessing settings,
so pass either full path to `evaluate.py --checkpoint` for later testing or labeled
batch inference. Exports happen before automatic test evaluation, so a test failure
does not prevent saving trained models. Set `training.saved_weights_dir: null` to
disable these extra copies; Python callers can inspect `trainer.saved_weights`.

The evaluation command restores architecture and preprocessing from the checkpoint
without accessing training files. Only load trusted local training checkpoints:
they contain Python/NumPy RNG state and use PyTorch's full checkpoint loader.
Pass `--quiet` to `evaluate.py` to print only the final JSON report.

## Python interface

```python
from fdmambadr import build_model
from utils.runtime import seed_everything

seed_everything(42)  # Before constructing a model.
model = build_model(
    backbone="mobilenetv3", variant="B3", fdconv={"kernel_num": 4}, mamba={"hidden_dim": 64}
)
print(model.build_config)  # Includes the resolved FDConv path.
```

| Variant | Existing constructor | Method |
|---|---|---|
| B0 | `Backbone` | Native CNN and native classifier |
| B1 | `FDConvBackbone` | One eligible backbone convolution replaced by FDConv |
| B2 | `MambaBackbone` | CNN plus local/global Mamba fusion |
| B3 | `HybridBackbone` | B1 plus exactly the B2 fusion block |

All variants support `mobilenetv3`, `efficientnet_b0`, and `resnet50`. Existing
constructors and imports remain available. `build_model` uses the native head;
the optional `BackboneWrapper` retains its configurable dense-head API:

```python
from fdmambadr.backbone import Backbone
from fdmambadr.wrapper import BackboneWrapper

model = BackboneWrapper(Backbone, name="mobilenetv3", hidden_dims=(256,), dropout=0.2)
```

For custom heads or datasets, use `Trainer` directly. The command-line evaluator
supports native-head checkpoints created with `build_model` and saved preprocessing;
custom architectures use their original constructor plus `Trainer.evaluate`.
`Trainer.fit()` returns the epoch history and stores automatic test results in
`trainer.test_report`. Pass `fit(evaluate_test=False)` to defer test evaluation;
set `verbose=False` when constructing the trainer to hide progress and summaries.
See [architecture and extension guide](docs/ARCHITECTURE.md).

## Code map

| Location | Responsibility |
|---|---|
| `fdmambadr/factory.py` | One variant-selection interface |
| `fdmambadr/backbone.py` | Native timm backbone construction |
| `fdmambadr/fdconv_models.py` | Selective adapter and FP32 execution wrapper |
| `fdmambadr/FDConv.py` | Protected supplied implementation; unchanged |
| `fdmambadr/mamba_fusion.py` | Shared local/Mamba/gate/residual block |
| `fdmambadr/fusion_backbone.py` | Shared B2/B3 forward interface |
| `utils/loaders.py`, `visualization.py` | Data loading and optional image grids |
| `utils/losses.py`, `metrics.py` | Losses and whole-split measurements |
| `utils/engine.py` | Batch processing, accumulation and optimizer updates |
| `utils/trainer.py` | Run lifecycle, checkpoints, selection and evaluation |
| `utils/runtime.py`, `data_checks.py` | Seeding, scheduling, identities and validation |
| `train.py`, `evaluate.py`, `configs/train.yaml` | Minimal executable workflow |

Keep weights in FP32; use autocast for AMP, not `model.half()` for FDConv. GPU AMP
and real Mamba execution remain pending. Exact epoch-resume replay is tested with
`num_workers=0`; persistent worker RNG states are not checkpointed.

These remain preliminary experiments under publisher-provided splits. The archive
contains no real dataset or trained results. CSV dataset adapters, comprehensive
duplicate/patient audits, experiment-suite orchestration, benchmarks and report
plots remain future work. See the review for details; no accuracy or generalization
improvement is claimed.

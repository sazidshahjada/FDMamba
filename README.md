# FDMamba

Medical image classification with MobileNetV3-Large, EfficientNet-B0 and ResNet-50.
All models train from scratch. The CNN, FDConv and Mamba computations are preserved;
the final classifier size comes from the chosen dataset.

For all **48 runs in baseline → FDConv → Mamba → hybrid order**, follow the
[step-by-step training guide](TRAINING_GUIDE.md). It lists every command for all
three backbones and four datasets, plus resume, inference and benchmarking.

## Setup

For the supplied Python 3.11 / PyTorch 2.5.1 / CUDA 12.4 environment:

```bash
conda env create -f environment.yml
conda activate FDMamba
```

Or, in a Python 3.11 virtual environment on Linux x86_64 / WSL2:

```bash
python -m pip install -r requirements.txt
```

For an existing environment with compatible PyTorch/timm/Mamba already installed:

```bash
python -m pip install medmnist==3.0.2 tensorboard==2.18.0
```

B2/B3 need CUDA and the real Mamba extensions. B0/B1 also work with a compatible CPU
PyTorch/torchvision installation. Keep model weights in FP32; `training.amp` controls
autocast. The full dependency files include CUDA builds and matching extension wheels.

## 1. Download data

```bash
python download_data.py
# Or choose datasets and native resolution:
python download_data.py --datasets pathmnist bloodmnist organamnist --size 28
```

MedMNIST itself downloads, verifies and reuses archives in `data/medmnist/`.
Each archive contains the official train, validation and test splits. Download
progress is provided by MedMNIST, with an outer progress bar for the dataset list.

| Dataset | Modality | Classes |
|---|---|---:|
| pathmnist | Histopathology | 9 |
| dermamnist | Dermatoscopy | 7 |
| bloodmnist | Blood-cell microscopy | 8 |
| organamnist | Abdominal CT slices | 11 |

`data.size` selects the official archive: 28, 64, 128 or 224. `data.image_size` is
the resized model input. Defaults are **28 → 224**, not native MedMNIST-224.
Larger archives need more disk and RAM. Grayscale images become RGB; labels retain
their official numeric order. Other single-label MedMNIST2D datasets also work.
Multilabel ChestMNIST and 3D volumes are outside this architecture's scope.

## 2. Train

```bash
python train.py --config configs/train.yaml --dataset pathmnist
python train.py --config configs/efficientnet.yaml --dataset bloodmnist
python train.py --config configs/resnet.yaml --dataset organamnist
```

Each YAML is a complete, readable recipe with `model`, `data` and `training`
sections. There is no configuration inheritance. Select an ablation or a custom
experiment name from the command line:

```bash
python train.py --config configs/train.yaml --dataset dermamnist --variant B0 --name MobileNet-baseline
python train.py --config configs/train.yaml --dataset dermamnist --variant B3 --name MobileNet-v1
python train.py --config configs/train.yaml --dataset dermamnist --variant B3 --name MobileNet-v1 --resume
```

`--backbone` accepts `mobilenetv3`, `efficientnet_b0` or `resnet50`.
`--device cpu`, `--device cuda:0` and `--download` are optional. Without
`--download`, training uses the existing MedMNIST cache. `download_data.py` verifies
completed archives, reuses them without network access, and resumes interrupted
transfers from `.npz.part` files (including partial files left by MedMNIST itself).

| Variant | Architecture |
|---|---|
| B0 | Native CNN baseline |
| B1 | CNN with one FDConv replacement |
| B2 | CNN with local/global Mamba fusion |
| B3 | Full FDMamba: FDConv + the same fusion |

Names are `FDMamba-MobileNetV3-B3`, `FDMamba-EfficientNetB0-B3`, and
`FDMamba-ResNet50-B3`, with the suffix changing for ablations.

Training uses AdamW, warm-up/cosine scheduling and cross-entropy. Class weights
come from training labels when `class_weighted: true`. Validation macro-F1 selects
`best.pt` by default; alternatives are accuracy, AUC, balanced accuracy and loss.
Exact ties retain the earlier checkpoint. The official test split is loaded only
after training ends and is evaluated using the selected checkpoint.

A default run writes:

```text
experiments/pathmnist/FDMamba-MobileNetV3-B3/size28_input224_seed42/
  config.yaml
  metadata.json
  best.pt
  last.pt
  history.json
  history.csv
  test_metrics.json
  test_predictions.csv
  tensorboard/events.out.tfevents.*
```

Set `training.output_dir` to override the location. Resume with the same total
epoch budget and settings; keep `best.pt` alongside `last.pt`. A fresh run refuses
to overwrite an existing experiment. Exact replay is tested with `num_workers: 0`.
For a one-epoch smoke test, set `epochs: 1`, `warmup_epochs: 0` and choose B0/B1.

After completion or early stopping, training also exports the validation-selected
model to `saved_weight/` (set `training.weights_dir` to change this):

```text
saved_weight/FDMamba-MobileNetV3-B3_pathmnist_size28_input224_seed42.pt
```

Names include the model, variant, dataset, resolution and seed. A custom `--name`
is added after the dataset. Exports contain weights, class names and preprocessing
for direct inference/benchmark use; resumable training state stays in the experiment
directory. A fresh run refuses to overwrite an existing export. Run the same
training command with `--resume` on a completed experiment to regenerate its export
without additional training. If the validation monitor is always undefined, the
last checkpoint is exported.

**TensorBoard is enabled for every run**, including resumed runs. It logs training,
validation and test scalars, learning rate, configuration and metadata:

```bash
tensorboard --logdir experiments
```

## 3. Classify an image

```bash
python inference.py \
  --model FDMamba-MobileNetV3-B3 \
  --checkpoint saved_weight/FDMamba-MobileNetV3-B3_pathmnist_size28_input224_seed42.pt \
  --image /path/to/image.png \
  --top-k 3 --output results/prediction.json
```

The checkpoint restores the architecture, class names, resolution and normalization.
`--model` is optional; when given, it checks the selected backbone, full model name,
or saved custom experiment name. JSON output contains class labels and probabilities.
Use images appropriate to the checkpoint's task and source preprocessing.

## 4. Benchmark saved checkpoints

Edit `configs/benchmark.yaml` with your names, checkpoint paths and datasets:

```bash
python benchmark.py --config configs/benchmark.yaml
```

Or provide matching lists directly:

```bash
python benchmark.py \
  --names MobileNet-v1 ResNet-v1 \
  --checkpoints /path/to/mobile/best.pt /path/to/resnet/best.pt \
  --datasets pathmnist organamnist \
  --output-dir results/comparison
```

Each name is a custom result label. The checkpoint determines the actual model.
Its dataset and label order must match the requested dataset. `--split val`,
`--data-root`, `--batch-size`, `--num-workers` and `--device` are also available.

Outputs include `summary.csv`, `summary.json`, and a metrics/predictions pair for
each named model. Reports contain accuracy, AUC, macro-F1, balanced accuracy,
per-class metrics, confusion matrices, parameter counts and evaluation throughput.
Timing includes image loading, preprocessing and forward passes; it is not a
controlled model-only latency measurement. Compare results within each dataset.
No training is performed by the benchmark command.

## Code map

```text
FDMamba/       Core CNN wrappers, selective FDConv and Mamba fusion
utils/         Data, config, epoch loop, training, metrics and evaluation helpers
configs/       Three training recipes and one checkpoint comparison example
saved_weight/  Named inference exports created after training
tests/         Offline regression tests
```

```python
from FDMamba import build_model

model = build_model(backbone="mobilenetv3", variant="B3", num_classes=9)
```

The source package and imports are now **`FDMamba`**, case-sensitive. The current
scripts use a compact checkpoint format created by the new `train.py`. Model
parameter paths remain unchanged, but older full trainer checkpoints are not
accepted for automatic resume. Only load trusted local checkpoints.

```bash
python -m unittest discover -s tests -v
```

See [architecture](docs/ARCHITECTURE.md), [verification](docs/IMPLEMENTATION_STATUS.md)
and [research scope](project_overview.md). MedMNIST metadata and AUC conventions
follow the [official project](https://github.com/MedMNIST/MedMNIST).
Default augmentation is off; transformations such as flips can alter organ
laterality. Fixed normalization must be shared by all splits. DermaMNIST is
CC BY-NC 4.0; the other selected datasets use CC BY 4.0. Preserve source attribution.

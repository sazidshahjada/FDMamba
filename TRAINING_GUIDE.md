# Train every FDMamba model, variant and dataset

Run these commands from the repository root, one at a time. Finish each stage
before starting the next. There are **48 independent training runs**:

| Stage | Variant | Architecture | Runs |
|---|---|---|---:|
| 1 | B0 | Original CNN baseline | 12 |
| 2 | B1 | CNN + FDConv | 12 |
| 3 | B2 | CNN + local/global Mamba fusion | 12 |
| 4 | B3 | Hybrid: CNN + FDConv + Mamba fusion | 12 |

Each stage trains MobileNetV3-Large, EfficientNet-B0 and ResNet-50 on PathMNIST,
DermaMNIST, BloodMNIST and OrganAMNIST. Every run starts from scratch; a later
variant does not load the earlier variant's weights. The architecture is unchanged.
In `EfficientNet-B0`, B0 is part of the backbone's name; `--variant B0/B1/B2/B3`
selects the experimental architecture independently.

## Prepare the environment

If you have not created the supplied environment yet:

```bash
conda env create -f environment.yml
```

Activate it before running the remaining commands:

```bash
conda activate FDMamba
```

If you already installed the dependencies into another environment, activate that
environment instead. Check your device:

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

B0/B1 can run on CPU or CUDA. B2/B3 require CUDA and the real Mamba extensions
included in the supplied environment. All commands below use `device: auto` from
the configs. Keep weights in FP32; `training.amp` controls optional autocast.

## Download all four datasets once

```bash
python download_data.py --datasets pathmnist dermamnist bloodmnist organamnist --size 28
```

MedMNIST downloads and verifies the archives with progress, then reuses them from
`data/medmnist/`. Every archive contains its official train, validation and test
splits. Subsequent training commands use the local cache without downloading again.

The provided recipes use native **28 × 28 data resized to 224 × 224**. To train
with native 224 × 224 data instead, download with `--size 224` and set `data.size:
224` in **all three** training configs before starting. Leave `data.image_size:
224`. Higher resolutions require more RAM and disk. Keep the same data resolution
and preprocessing across variants when comparing them.

## Review the three recipes

| Backbone | Config | Maximum epochs | Batch size | Accumulation | Learning rate |
|---|---|---:|---:|---:|---:|
| MobileNetV3-Large | `configs/train.yaml` | 100 | 32 | 1 | 0.001 |
| EfficientNet-B0 | `configs/efficientnet.yaml` | 120 | 16 | 2 | 0.00075 |
| ResNet-50 | `configs/resnet.yaml` | 150 | 8 | 4 | 0.0005 |

The commands explicitly select the variant, overriding `model.variant` in YAML.
The dataset determines the output classes automatically: 9, 7, 8 and 11,
respectively. The default seed is 42, augmentation is off, and validation macro-F1
selects the best checkpoint. Early stopping may finish before the maximum epoch.

Edit hyperparameters before starting the experiments. Leave `training.output_dir:
null` to keep every model/variant/dataset in its own experiment directory. Leave
`training.weights_dir: saved_weight` for the export paths used throughout this guide.
If GPU memory is insufficient, lower `data.batch_size`; increasing
`training.accumulation_steps` can preserve the number of samples per optimizer
update, although batch normalization still sees the smaller batch. Use the same
recipe for all variants of a backbone for the intended comparison.

Open TensorBoard in a second terminal using the same environment:

```bash
tensorboard --logdir experiments
```

Open the local URL printed by TensorBoard. Logging is automatic for every run,
including resumed runs. Each experiment has its own `tensorboard/` directory.

## Stage 1: train all baselines — B0

### MobileNetV3-Large, runs 01–04

```bash
python train.py --config configs/train.yaml --variant B0 --dataset pathmnist
python train.py --config configs/train.yaml --variant B0 --dataset dermamnist
python train.py --config configs/train.yaml --variant B0 --dataset bloodmnist
python train.py --config configs/train.yaml --variant B0 --dataset organamnist
```

### EfficientNet-B0, runs 05–08

```bash
python train.py --config configs/efficientnet.yaml --variant B0 --dataset pathmnist
python train.py --config configs/efficientnet.yaml --variant B0 --dataset dermamnist
python train.py --config configs/efficientnet.yaml --variant B0 --dataset bloodmnist
python train.py --config configs/efficientnet.yaml --variant B0 --dataset organamnist
```

### ResNet-50, runs 09–12

```bash
python train.py --config configs/resnet.yaml --variant B0 --dataset pathmnist
python train.py --config configs/resnet.yaml --variant B0 --dataset dermamnist
python train.py --config configs/resnet.yaml --variant B0 --dataset bloodmnist
python train.py --config configs/resnet.yaml --variant B0 --dataset organamnist
```

## Stage 2: train all FDConv models — B1

### MobileNetV3-Large + FDConv, runs 13–16

```bash
python train.py --config configs/train.yaml --variant B1 --dataset pathmnist
python train.py --config configs/train.yaml --variant B1 --dataset dermamnist
python train.py --config configs/train.yaml --variant B1 --dataset bloodmnist
python train.py --config configs/train.yaml --variant B1 --dataset organamnist
```

### EfficientNet-B0 + FDConv, runs 17–20

```bash
python train.py --config configs/efficientnet.yaml --variant B1 --dataset pathmnist
python train.py --config configs/efficientnet.yaml --variant B1 --dataset dermamnist
python train.py --config configs/efficientnet.yaml --variant B1 --dataset bloodmnist
python train.py --config configs/efficientnet.yaml --variant B1 --dataset organamnist
```

### ResNet-50 + FDConv, runs 21–24

```bash
python train.py --config configs/resnet.yaml --variant B1 --dataset pathmnist
python train.py --config configs/resnet.yaml --variant B1 --dataset dermamnist
python train.py --config configs/resnet.yaml --variant B1 --dataset bloodmnist
python train.py --config configs/resnet.yaml --variant B1 --dataset organamnist
```

## Stage 3: train all Mamba models — B2

### MobileNetV3-Large + Mamba, runs 25–28

```bash
python train.py --config configs/train.yaml --variant B2 --dataset pathmnist
python train.py --config configs/train.yaml --variant B2 --dataset dermamnist
python train.py --config configs/train.yaml --variant B2 --dataset bloodmnist
python train.py --config configs/train.yaml --variant B2 --dataset organamnist
```

### EfficientNet-B0 + Mamba, runs 29–32

```bash
python train.py --config configs/efficientnet.yaml --variant B2 --dataset pathmnist
python train.py --config configs/efficientnet.yaml --variant B2 --dataset dermamnist
python train.py --config configs/efficientnet.yaml --variant B2 --dataset bloodmnist
python train.py --config configs/efficientnet.yaml --variant B2 --dataset organamnist
```

### ResNet-50 + Mamba, runs 33–36

```bash
python train.py --config configs/resnet.yaml --variant B2 --dataset pathmnist
python train.py --config configs/resnet.yaml --variant B2 --dataset dermamnist
python train.py --config configs/resnet.yaml --variant B2 --dataset bloodmnist
python train.py --config configs/resnet.yaml --variant B2 --dataset organamnist
```

## Stage 4: train all hybrid models — B3

### MobileNetV3-Large + FDConv + Mamba, runs 37–40

```bash
python train.py --config configs/train.yaml --variant B3 --dataset pathmnist
python train.py --config configs/train.yaml --variant B3 --dataset dermamnist
python train.py --config configs/train.yaml --variant B3 --dataset bloodmnist
python train.py --config configs/train.yaml --variant B3 --dataset organamnist
```

### EfficientNet-B0 + FDConv + Mamba, runs 41–44

```bash
python train.py --config configs/efficientnet.yaml --variant B3 --dataset pathmnist
python train.py --config configs/efficientnet.yaml --variant B3 --dataset dermamnist
python train.py --config configs/efficientnet.yaml --variant B3 --dataset bloodmnist
python train.py --config configs/efficientnet.yaml --variant B3 --dataset organamnist
```

### ResNet-50 + FDConv + Mamba, runs 45–48

```bash
python train.py --config configs/resnet.yaml --variant B3 --dataset pathmnist
python train.py --config configs/resnet.yaml --variant B3 --dataset dermamnist
python train.py --config configs/resnet.yaml --variant B3 --dataset bloodmnist
python train.py --config configs/resnet.yaml --variant B3 --dataset organamnist
```

## Locate the exported model weights

Each run exports one `.pt` file automatically when training completes or early
stops. After all 48 runs, the default `saved_weight/` directory contains 48 exports:

```text
saved_weight/
  FDMamba-MobileNetV3-B0_pathmnist_size28_input224_seed42.pt
  FDMamba-MobileNetV3-B1_pathmnist_size28_input224_seed42.pt
  FDMamba-MobileNetV3-B2_pathmnist_size28_input224_seed42.pt
  FDMamba-MobileNetV3-B3_pathmnist_size28_input224_seed42.pt
  FDMamba-EfficientNetB0-B0_dermamnist_size28_input224_seed42.pt
  FDMamba-ResNet50-B3_organamnist_size28_input224_seed42.pt
  ...
```

The naming rule is:

```text
FDMamba-{backbone}-{variant}_{dataset}_size{archive_size}_input{image_size}_seed{seed}.pt
```

Backbone names in filenames are `MobileNetV3`, `EfficientNetB0` and `ResNet50`.
Variants are `B0`, `B1`, `B2` and `B3`. The filename separates different seeds and
resolutions. The export contains **validation-selected weights**, their epoch,
model configuration, dataset labels and preprocessing settings. By default,
selection uses the highest validation macro-F1; if the configured monitor is
undefined for the entire run, the last checkpoint is exported. The test split is
evaluated after selection and does not choose the model.

These compact exports work directly with `inference.py` and `benchmark.py`.
Optimizer state and training history remain in the experiment checkpoints:

```text
experiments/pathmnist/FDMamba-MobileNetV3-B0/size28_input224_seed42/
  best.pt                 # Best validation checkpoint, including training state
  last.pt                 # Latest resumable training checkpoint
  config.yaml
  metadata.json
  history.json
  history.csv
  test_metrics.json
  test_predictions.csv
  tensorboard/
```

## Resume, repeat, or export an existing completed run

If training is interrupted, repeat its exact command with `--resume`. For example:

```bash
python train.py --config configs/train.yaml --variant B0 --dataset pathmnist --resume
```

Resume restores the last completed epoch, optimizer, scheduler and random states.
Keep the same dataset, variant, seed, total epoch budget and other training
settings. Keep `best.pt` alongside `last.pt` when a best checkpoint exists. An
interruption before the first checkpoint cannot resume; use a new `--name`.

For an already completed run, the same `--resume` command re-exports its selected
weights and repeats test evaluation without training more epochs. This also works
for existing version-2 training checkpoints made before automatic exports were
added. `training.weights_dir` may be changed when relocating exports. Compact
files in `saved_weight/` cannot resume training; retain the experiment directory.

To repeat an experiment with different settings, choose a new name:

```bash
python train.py --config configs/train.yaml --variant B0 --dataset pathmnist --name repeat1
```

The export then includes the custom name after the dataset:

```text
saved_weight/FDMamba-MobileNetV3-B0_pathmnist_repeat1_size28_input224_seed42.pt
```

A fresh run refuses to overwrite an existing experiment or export. Changing the
seed in YAML gives a separate experiment and export automatically. If you set
`training.output_dir` explicitly, assign a distinct folder for every fresh run.

## Run inference using saved weights

After the first baseline finishes, replace the image path with a PathMNIST-task
image and run:

```bash
python inference.py \
  --model FDMamba-MobileNetV3-B0 \
  --checkpoint saved_weight/FDMamba-MobileNetV3-B0_pathmnist_size28_input224_seed42.pt \
  --image /path/to/image.png \
  --top-k 3 \
  --output results/pathmnist_prediction.json
```

For a hybrid model, select its B3 file and a CUDA device:

```bash
python inference.py \
  --model FDMamba-MobileNetV3-B3 \
  --checkpoint saved_weight/FDMamba-MobileNetV3-B3_pathmnist_size28_input224_seed42.pt \
  --image /path/to/image.png \
  --device cuda:0 \
  --top-k 3
```

The file restores the model, class names and preprocessing; no training config
argument is needed. `--model` is optional and verifies the model selection.

## Benchmark the trained variants

Compare all four MobileNetV3 variants on PathMNIST:

```bash
python benchmark.py \
  --names MobileNet-B0 MobileNet-B1 MobileNet-B2 MobileNet-B3 \
  --checkpoints \
    saved_weight/FDMamba-MobileNetV3-B0_pathmnist_size28_input224_seed42.pt \
    saved_weight/FDMamba-MobileNetV3-B1_pathmnist_size28_input224_seed42.pt \
    saved_weight/FDMamba-MobileNetV3-B2_pathmnist_size28_input224_seed42.pt \
    saved_weight/FDMamba-MobileNetV3-B3_pathmnist_size28_input224_seed42.pt \
  --datasets pathmnist pathmnist pathmnist pathmnist \
  --device cuda:0 \
  --output-dir results/pathmnist_mobilenet
```

The names are your result labels; each checkpoint determines the actual model.
To compare another backbone or dataset, replace the checkpoint paths, dataset
arguments and output directory together. Each checkpoint must use the dataset it
was trained on. Compare performance within the same dataset.

You can also edit the list in `configs/benchmark.yaml` and run:

```bash
python benchmark.py --config configs/benchmark.yaml
```

Every entry has a unique `name`, a `checkpoint` path and its `dataset`. Add as many
entries as needed, including all 48 exports. Benchmarking writes `summary.csv`,
`summary.json` and per-model metrics/predictions under the selected results folder.
Reported metrics include accuracy, AUC, macro-F1 and balanced accuracy. Throughput
includes loading and preprocessing, so it is not an isolated model latency test.

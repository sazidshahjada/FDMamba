# Exported model weights

Training automatically writes one compact `.pt` checkpoint here after completion
or early stopping. The default filename includes model, variant, dataset, native
data resolution, input resolution and seed:

```text
FDMamba-MobileNetV3-B0_pathmnist_size28_input224_seed42.pt
FDMamba-EfficientNetB0-B1_bloodmnist_size28_input224_seed42.pt
FDMamba-ResNet50-B3_organamnist_size28_input224_seed42.pt
```

An optional `train.py --name repeat1` adds `_repeat1` after the dataset. Change
`training.weights_dir` in YAML to choose another export directory.

Exports contain the validation-selected weights plus the model configuration,
class names and preprocessing needed by `inference.py` and `benchmark.py`. If the
validation monitor is always undefined, the last checkpoint is used. Full training
state stays in `experiments/`; use that directory and `train.py --resume` to resume
or regenerate an export from a completed run.

Generated weights are ignored by Git. See the [full training guide](../TRAINING_GUIDE.md)
for the 48 training commands, saved-file names, inference and benchmarking.

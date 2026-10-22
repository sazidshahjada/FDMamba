# Implementation verification — 23 October 2026

The repository now has one `FDMamba` model package and four root entry points:
`train.py`, `download_data.py`, `inference.py`, and `benchmark.py`. Training recipes
are plain YAML. Data loading and download/cache behavior use MedMNIST directly.
TensorBoard is mandatory for every training run and writes into that run's folder.
Completed and early-stopped runs also export validation-selected weights to
`saved_weight/`. The [training guide](../TRAINING_GUIDE.md) lists all 48 commands in
B0 → B1 → B2 → B3 order across three backbones and four datasets.

## Architecture checks

Before reorganizing the models, six B0/B1 reference states and outputs were saved
for all three backbones under the same seed. After the rewrite, parameter keys,
initialized values and CPU outputs were exactly equal (`rtol=0`, `atol=0`).

Constructor and forward implementations for Backbone, FDConvBackbone, FP32FDConv,
FusionBackbone, MambaBackbone, HybridBackbone and LocalGlobalMambaFusion were also
compared structurally. Computation is unchanged; package paths and docstrings were
updated. The supplied FDConv received four concise class docstrings; its executable
statements are unchanged, including the earlier standalone compatibility fixes
listed in [ARCHITECTURE.md](ARCHITECTURE.md).

The unused dense-head wrapper, duplicate package aliases, generic trainer hooks,
custom downloader, folder-dataset adapters and redundant preset files were removed.
Current scripts use a compact version-2 checkpoint containing weights, model/data
settings, label names and resumable training state. Old full trainer checkpoint
formats are not automatically resumed. Existing architecture state-dict paths
remain unchanged.

## Tests

Environment: Python 3.11.17, PyTorch 2.5.1, torchvision 0.20.1, timm 0.9.16,
MedMNIST 3.0.2. CUDA is not visible in this session.

```bash
python -m unittest discover -s tests -v
```

**11 tests: 10 passed, 1 CUDA-only test skipped.** Subtests cover all three B0/B1
backbones and the four default datasets. Synthetic archives are created only in
temporary directories; no network access or real benchmark scores are required.

Verified:

- Direct official dataset classes, cached download reuse, numeric label order,
  grayscale conversion, deterministic held-out transforms and scalar targets.
- Rejection of unsupported 3D/multilabel datasets and incompatible settings.
- B0/B1 forward/backward and active FDConv spectral gradients. ResNet's preserved
  zero-initialized residual scale needs its first optimizer update before checking
  gradients farther back in that branch.
- Weighted gradient accumulation, including an incomplete final window, agrees
  with corresponding full-window cross-entropy updates.
- Binary/multiclass AUC matches MedMNIST; absent classes yield explicit undefined AUC.
- Epoch-boundary resume matches uninterrupted training exactly with zero workers.
- TensorBoard files contain training, validation and test metrics, learning rate
  and both epochs after resume, with no duplicate training steps.
- Single-image inference matches the corresponding dataset prediction after saved
  preprocessing; incompatible model/dataset selections are rejected.
- Named checkpoint benchmarking writes comparison CSV/JSON and per-model predictions.
- Compact exports contain the exact selected weights and reconstruction metadata,
  omit optimizer/history state, and load directly for inference and benchmarking.
- Exports are not written for incomplete runs. Early stopping exports the last
  checkpoint when validation AUC is undefined; completed runs can regenerate
  exports in a different directory without additional training.
- Model/variant/dataset/size/seed filenames and custom run names keep outputs
  distinct. Fresh training refuses to overwrite an existing export.

Ruff lint/format, Python syntax, CLI help, documentation links and stale-import
checks passed alongside the regression suite.

## Remaining limits

No full training or scientific benchmark has been performed. Real CUDA Mamba and
GPU AMP remain unverified here. The conditional GPU test is available on the
intended environment; no surrogate Mamba is used.

Full dataset downloads remain the user's normal preparation step. An old partial
archive from an earlier transfer remains under `data/medmnist/`; MedMNIST does not
use custom `.part` files. `download_data.py` follows the official module's download
and retry behavior. Dataset source resolution and model input resolution remain
separate settings; defaults are 28 → 224.

Evaluation timing includes data loading/preprocessing and inference, without a
controlled warm-up protocol. It is useful run throughput, not isolated model
latency. Clinical performance, patient-level independence, duplicate audits and
performance improvements are not established by these software checks.

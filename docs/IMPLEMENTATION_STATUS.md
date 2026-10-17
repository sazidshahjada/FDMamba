# Implementation review — 17 October 2026

Reviewed `project_overview.md` and all five existing Python modules. The repository
contained model code and two external annotation CSVs; it contained no training,
evaluation, dataset-loader, or checkpoint-management implementation. No existing
`AGENTS.md`, prior status document, supplied MambaLiteUNet archive, or original
FDConv backup was present.

## Corrections

| Finding | Resolution |
|---|---|
| `fdconv_models.py` imported nonexistent `.backbones` | Import the existing `.backbone`. Correct stale paths/documentation. |
| MobileNetV3 spatial features have 960 channels, while `num_features` on its timm model reports the 1280-channel classifier embedding | The wrapper exposes the channel count before `conv_head`; B2 and B3 use that shared resolution. EfficientNet and ResNet remain 1280 and 2048. |
| B3 added a second FDConv in its fusion branch, changing two components compared with B2 | B3 now composes the B1 adapter with the exact B2 fusion class. Removed `FDConvLocalGlobalFusion`, `local_kernel_num`, and `local_fdconv_kwargs`. Old hybrid states/configs are incompatible and require a fresh run. |
| Eager Mamba imports blocked module inspection without CUDA extensions | Import Mamba on fusion construction. B2/B3 still fail clearly if its real implementation is unavailable; CPU forwards are rejected explicitly. |
| B2 had different backbone-name validation and no classifier dropout option | Reuse baseline construction/aliases and expose `drop_rate` separately from fusion `dropout`. Validate supported Mamba settings. |
| Adapter could select SE pooled-descriptor convolutions or silently construct inactive FDConv | Exclude SE and unsupported padding sites; require active spectral parameters. Record padding/dilation, bias, and parameter counts with the selected layer. |
| FDConv required MMCV although no detection/segmentation registry was used | Remove registry import/decorator and unused NumPy/Matplotlib/timm imports. Retain the supplied FDW, global/local KSM, and FBM algorithms. |
| KSM global pooled BatchNorm failed during training with batch size one | Use its running statistics for a singleton descriptor, preserving affine gradients; larger batches retain standard BatchNorm. This is an explicit small-batch behavior choice. |
| FBM interpolated a fixed-size frequency mask, shifting cutoffs on smaller and odd grids | Recompute masks from the actual FFT bins when spatial dimensions change; cache the last size as a nonpersistent buffer. |
| Kernel inverse FFT inferred an even width and failed with odd channel/kernel products | Supply the original two-dimensional FFT shape. Also supply the original length in local KSM's optional 1-D inverse FFT. |
| `kernel_num=1` produced a scalar attention followed by a tensor reshape | Broadcast scalar kernel attention explicitly. |
| FDConv ignored its stride eligibility list | Honor stride eligibility and use ordinary convolution for ineligible cases; the backbone adapter continues to require active FDConv. |
| Several unsupported FDConv configurations failed inside tensor reshaping or silently lost options | Reject invalid kernel partition counts, grouped/nonsquare active kernels, nonzero padding modes, invalid temperatures/reductions, spatial frequency decomposition, unsupported attention grids, and incompatible softmax modulation. |
| Filter attention initialization checked a bound method instead of the filter convolution | Apply the intended small initialization to `filter_fc`. |
| Checkpoint mode omitted an explicit PyTorch reentrancy choice | Use `use_reentrant=False`. |
| `profile_module` referenced undefined attributes and returned an unvalidated estimate | Raise a descriptive `NotImplementedError`; use runtime profiling and disclose missing FFT/dynamic-kernel MAC accounting. |

No pretrained weights are enabled. The FDConv layer receives a fresh random
spatial initialization before Fourier conversion. Shared backbone layers are not
copied from a trained model. No accuracy, clinical, or generalization claims have
been tested.

## Resolved architecture choices

B2 and B3 share one projected feature tensor between their local depthwise/PW
convolution and Mamba branches, as the original B2 implementation did. Fusion is
after `forward_features`, before the original classification head. This retains
the existing shared-projection adaptation of the overview's schematic, which
depicts separate branch projections. Both branches use raster ordering and a
learned residual scale initialized to 0.01.

With timm **0.9.16**, default B1/B3 replacement selection resolves to:

| Backbone | Path relative to timm model | Channels | Kernel | FBM |
|---|---|---|---|---|
| MobileNetV3-Large | `blocks.2.2.conv_pwl` | 120 → 40 | 1×1 | Disabled by source defaults |
| EfficientNet-B0 | `blocks.2.0.conv_pwl` | 144 → 40 | 1×1 | Disabled by source defaults |
| ResNet-50 | `layer3.3.conv2` | 256 → 256 | 3×3 | Enabled |

MobileNet/EfficientNet use ungrouped pointwise operations because their spatial
operations are depthwise; this is a comparison limitation. FDW and KSM are active
at all three sites. Auto-selection depends on timm version and configuration;
save `replacement` metadata and freeze an explicit `layer_path` for actual runs.

## Validation

Available execution environment: Python **3.12.12**, PyTorch **2.11.0+cu130**,
timm **0.9.16**, **CPU**. No CUDA device, CUDA compiler, `mamba_ssm`, or
`causal_conv1d` is available here. This is distinct from the proposed GPU YAML.

| Check | Result |
|---|---|
| Parse all model/test Python files; parse environment YAML | Passed |
| Conda dry-run dependency solve of `environment.yml` | Passed; resolves CUDA PyTorch 2.5.1, CUDA torchvision 0.20.1, torchtriton 3.1.0 and Python 3.11 |
| `python -m unittest discover -s tests -v` | 13 tests: 11 passed, 2 CUDA tests skipped |
| B0 and B1, all three backbones, `[1,3,224,224]` | Five logits, finite loss/gradients, two AdamW steps, evaluation forward passed |
| FDConv spectral/global KSM/local KSM/FBM gradients and upstream gradients | Passed isolated tests; spectral gradient is nonzero after the second model step |
| Odd FFT widths, one kernel, batch sizes 1 and 2 | Passed |
| FBM frequency decomposition compared with directly constructed FFT masks | Passed multiple odd/even grids |
| Spatial vs spectral parameterization and serialized state round trip | Passed |
| CPU bfloat16 autocast with FP32 FDConv internals | Passed |
| Invalid replacement sites and silently inactive FDConv | Rejected as expected |
| B2/B3 real CUDA/selective scan, GPU AMP, GPU optimizer steps | **Pending GPU verification**; tests included, no surrogate Mamba used |
| Full environment installation and import of its binary extensions | **Not performed**; no GPU available |

ResNet's standard timm initialization zeros the last BatchNorm scale in residual
branches. Some convolution gradients are therefore initially zero; the tests
verify nonzero FDConv spectral gradients after that scale's first optimizer
update, without changing the baseline initialization policy.

The YAML pins PyTorch/torchvision/CUDA and uses official CPython 3.11, Linux
x86_64, CUDA 12, torch 2.5, C++ ABI FALSE extension wheels. Their asset names were
verified against the official GitHub release metadata. This avoids compiling
Mamba against an isolated or mismatched PyTorch. It is not a cross-platform lock
file; export the fully resolved environment and hardware details on the training
machine. The Conda dry run checks Conda package resolution; it does not install or
execute the pip extension wheels. Dependencies for image processing, manifests, QWK/F1 metrics, plots,
TensorBoard, and resource monitoring are included.

## Configurable classifier wrapper

Added `fdmambadr/wrapper.py` with `BackboneWrapper`. Its first argument is a
backbone **class** (`Backbone`, `FDConvBackbone`, `MambaBackbone`, or
`HybridBackbone`), followed by the architecture name and output class count.
Hidden widths, per-layer/shared dropout, activation class and arguments,
normalization, linear biases, and backbone constructor options are configurable.

The wrapper removes the native final linear classifier and delegates spatial
feature processing and pooling/projection to the selected backbone. This keeps
MobileNet's native 1280-dimensional projected embedding separate from its
960-channel spatial map and avoids retaining unused classifier parameters.
The wrapper is optional; existing constructors retain their native heads.
It does not enable pretrained weights. Keep head settings identical across
paired ablations, and record `model.config` plus resolved backbone metadata.

Validation: `python -m unittest discover -s tests -p test_wrapper.py -v` ran
**5 tests: 4 passed, 1 CUDA test skipped**. Verified all three architectures with
both B0/B1 classes, batch-one forward/backward with finite gradients for every
retained parameter, varying spatial sizes, linear-only heads, custom class
counts, activation arguments, dense BatchNorm with batch two, grayscale inputs,
FDConv constructor options, and rejected invalid settings. Real B2/B3 wrapper
execution remains pending CUDA; a GPU integration test is included.

## ImageFolder loaders and grid visualization

Implemented `utils/loaders.py`:

- `create_dataloaders(train_path, val_path, test_path, ...)` returns loaders keyed
  by `train`, `val`, and `test`. Images are decoded as RGB, resized, and converted
  to float32 tensors. Only training is shuffled or optionally augmented. All
  validation/test examples are retained with deterministic transforms.
- Labels follow an explicit ordered class mapping (default DR folders `0..4`).
  Missing split classes preserve their indices; unexpected directories and empty
  datasets fail. Datasets expose discovered samples, labels and class counts.
- Optional normalization is shared across splits. `compute_train_mean_std` fits
  the resized training pixels only; no automatic all-dataset statistics scan or
  ImageNet normalization is applied. It reads all training images if invoked.
- Worker seeding, independent split generators, optional pinned memory, and
  persistent workers are supported. Zero workers is the portable default.
- `visualize_images(loaders, rows, cols, split, ...)` renders labeled grids,
  reverses normalization, supports saving, hides unused cells, and preserves
  CPU augmentation and loader shuffle RNG state.
- Equal/nested split roots are rejected. This does not replace duplicate,
  patient/provenance, symlinked-file, or full-image integrity auditing.

Validation: **7 synthetic loader tests passed**, including multi-worker loading,
fixed labels with absent classes, deterministic evaluation, full held-out sample
coverage, grayscale-to-RGB conversion, train-only pixel statistics, normalization,
grid saving/RNG preservation, and invalid inputs. The multi-worker test passed
outside the sandbox, which restricts local process communication.

A batch of two images from each real split of the on-disk
`augmented_resized_V2` collection loaded as finite `[2,3,224,224]` tensors with
valid labels. ImageFolder discovered **115,241 training**, **14,227 validation**,
and **14,201 test** image files. These are file counts, not verified independent
retinas; only the sampled batches were decoded in this smoke check. No full
training, duplicate/patient audit, or external evaluation was performed.

## Classification losses

Implemented `utils/losses.py` with:

- `CrossEntropy`: PyTorch-compatible hard-label cross-entropy with optional class
  weights, label smoothing, ignored labels and none/sum/mean reductions.
- `BalancedCrossEntropy`: fixed training inverse-frequency weights
  `N / (C * count[c])`. A count vector is accepted directly;
  `from_dataset(loaders['train'].dataset)` computes counts from existing labels
  without reading pixels. Absent training classes fail explicitly. The term
  "balanced" here denotes inverse-frequency weighting. Weights/counts are buffers.
- `FocalLoss`: multiclass softmax adaptation of Lin et al., with configurable
  nonnegative gamma and scalar/per-class alpha. The focusing probability comes
  from unweighted cross-entropy; alpha is applied afterwards. Its mean reduction
  averages over nonignored samples, while weighted CE uses target-weight sums.

Inputs are logits `[B,C]` and integer `torch.long` labels `[B]`. FP16/bfloat16
inputs use FP32 arithmetic. Empty/all-ignored batches return differentiable zero;
cross-entropy also returns zero for a zero-effective-weight batch. The focal
factor uses `expm1` and a tiny clamp to keep confident-prediction gradients finite
with fractional gamma. Training counts must never be fitted on held-out labels.

Validation: **11 loss tests: 10 passed, 1 CUDA test skipped**. Checked CE values
and gradients against PyTorch, all reductions, smoothing, class weights/counts,
dataset label ordering and buffer restoration, the independent focal formula,
gamma-zero equivalence, scalar alpha scaling, ignored labels, extreme logits,
FP16/bfloat16/FP32/FP64 focal gradients, double-precision gradient checks, and CPU
autocast/AdamW updates. Real GPU loss/AMP execution remains pending.

A real two-image training batch was also passed through `BackboneWrapper` with
MobileNetV3. All three losses and their classification-head gradients were finite;
balanced/focal weights used the training ImageFolder label counts only. This was
a forward/backward smoke check, not a completed training experiment.

References: [PyTorch CrossEntropyLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html),
[Lin et al., Focal Loss](https://arxiv.org/abs/1708.02002).

## Classification metrics

Implemented `utils/metrics.py` with `compute_metrics(predictions, targets)` and
`ClassificationMetrics.update/compute/reset` for epoch/split accumulation.
Reports include accuracy, balanced accuracy, macro precision/recall/F1, QWK,
per-grade precision/recall/F1/support, and a fixed-size confusion matrix with
true labels in rows. QWK uses actual ordinal grade distances, retaining missing
intermediate grades. Metrics are computed on the concatenated split, never by
averaging batch QWK/F1 values.

Logits, explicit probability matrices, and hard predictions are supported.
Tensor inputs are detached and moved to CPU; label order, shapes, finite scores,
probability normalization and bounds are checked. Ignored labels are excluded;
empty/all-ignored splits fail. Inputs are copied, reset clears accumulated data,
and `get_predictions()` supplies CPU arrays for future export scripts.

Macro scores include all configured grades with zero for undefined per-class
precision/recall/F1. Balanced accuracy averages recall over true classes present
in the split. QWK with zero expected disagreement returns `None` and a reason.
The JSON-serializable report contains no NaN; an undefined QWK must be handled
explicitly by the future checkpoint selector.

The exploratory referable-DR report uses true grade >=2 by default. With scores,
it sums grade 2–4 probabilities, applies a preselected threshold of 0.5 for
sensitivity/specificity, and computes AUC from the continuous sum. Hard-grade
inputs use the predicted-grade boundary and have no AUC. Missing true binary
groups produce `None` for the affected rates/AUC, with explicit reasons.
Boundary/threshold are configurable; no external threshold fitting occurs.

Validation: **9 metric tests: 8 passed, 1 CUDA test skipped**. Compared accuracy,
balanced accuracy, QWK, F1, per-class scores and confusion matrices against
scikit-learn; checked probability/logit equivalence, unequal-batch accumulation,
missing grades, single-class degeneracy, binary sensitivity/specificity/AUC,
summed-probability decisions, ignored/empty input, reset/copy isolation, invalid
inputs, strict JSON serialization and extreme/half/bfloat16 logits. Actual CUDA
tensor ingestion remains pending GPU verification. No trained-model performance
or external evaluation was measured.

References: [scikit-learn QWK](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.cohen_kappa_score.html),
[balanced accuracy](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html),
[ROC-AUC](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html).

## Remaining implementation and experimental work

- Implement portable manifests, dataset identities and CSV-driven external
  loaders; verify each source's grade mapping. Preserve the supplied split policy
  and deferred-audit caveats.
- Implement the common training loop, integrate the training-fitted loss, validation QWK
  selection, deterministic tie policy, early stopping, complete checkpoint/RNG
  resume, saved configurations, and prediction/metric exports.
- Implement frozen-checkpoint internal and per-source external evaluation,
  environment capture, runtime benchmarks, and the 8/12-run experiment launcher.
- On the GPU machine, create the YAML environment, run `pip check`, explicitly
  verify CUDA/extension imports, and run the full regression suite without skips.
- Perform dataset integrity checks and a short real-data run before full
  experiments. No dataset audit, full training, or external evaluation ran here.

Dependency references: [PyTorch 2.5.1 installation matrix](https://pytorch.org/get-started/previous-versions/#v251),
[Mamba 2.2.4 release](https://github.com/state-spaces/mamba/releases/tag/v2.2.4),
[causal-conv1d 1.5.4 release](https://github.com/Dao-AILab/causal-conv1d/releases/tag/v1.5.4).

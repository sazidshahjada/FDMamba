# Code review and revision — 17 October 2026

## Verdict

The uploaded implementation is broadly consistent with the current project plan.
Its default B0/B1 model, loss, metric and training behavior passes the available CPU
checks. This revision fixes orchestration/reproducibility gaps and makes the code
easier to use and extend. It does **not** establish correctness of real CUDA Mamba,
all optional FDConv configurations, the real dataset's labels/splits, or trained
model performance.

`fdmambadr/FDConv.py` is **byte-for-byte unchanged**, including its documentation.
Its SHA-256 before and after this revision is:

```text
000a82f7daa8b52ef02e3e6f839e7ed92ac6e04b4ed575ab2effedfd3d94af79
```

The supplied file already contained earlier standalone adaptations. Those are
preserved, not attributed to this revision. The original status file is retained
as `PRIOR_IMPLEMENTATION_NOTES.md`, explicitly marked historical.

## Findings and resolutions

| Finding in the uploaded archive | Revision |
|---|---|
| README/status claimed training and checkpoint support were unimplemented, although a substantial trainer and tests already existed. The documented Conda activation name also differed from the YAML. | Replaced stale operational documentation and corrected activation to `fdmambadr`. Historical notes are separate. |
| Model construction required choosing among four constructors with different FDConv/fusion keyword names. | Added one `build_model` factory with B0–B3 selection and explicit rejection of unused/inapplicable options. Existing constructors remain. |
| B0/B1/B2 lacked consistent constructor metadata; no general executable checkpoint reconstruction path existed. | Added model configuration records and `model.build_config`, including resolved FDConv paths; checkpoints retain the build specification. |
| Local/global fusion lived in the B2 model file; B2 and B3 duplicated their forward routing. | Extracted the shared fusion implementation and common forward interface without changing parameter paths or computation. |
| The trainer combined process RNG utilities, data metadata handling, batch execution, selection and persistence in one file. | Moved runtime helpers, dataset checks and the epoch execution base into focused modules. Kept the public trainer and its supported hooks. |
| A fully frozen model could not construct a trainer for evaluation because an optimizer with trainable parameters was mandatory. An existing test worked around this by unfreezing one weight. | Evaluation accepts fully frozen models with no optimizer; removed the workaround and added a regression. |
| Validation normalization was not compared with training at trainer construction; only final evaluation checked it. | Training/validation/test normalization metadata must match before fitting. |
| `seed_everything(..., deterministic=False)` left deterministic mode enabled after a prior `True` call. | Explicitly applies either setting and records the actual runtime mode. |
| Resuming into another directory could retain the historical best score without retaining its checkpoint, leaving `best.pt` unavailable if no new best occurred. | Require resume within the checkpoint run directory and require the historical `best.pt` when a prior best exists. Relocate the complete run directory. |
| Temporary checkpoint evaluation copied the current model state on its current device, adding an avoidable model-sized GPU allocation. | Save the temporary state on CPU and restore it after evaluation or errors. This trades host memory for lower device-memory demand. |
| There was no concise runnable train/evaluate workflow; training-folder construction required a test path. | Added `train.py`, `evaluate.py`, one example YAML, optional test-path loading and `create_eval_loader`. Training and test execution are separate. |
| Formatting and documentation were inconsistent, with optional plotting mixed into data loading. | Applied consistent formatting/import cleanup outside FDConv, added concise responsibility-level docstrings, and separated plotting while preserving its old import. Ruff explicitly excludes FDConv. |

## Methodological choices retained

- Three scratch-initialized timm backbones; no pretrained weights or downloads.
- One selected active FDConv replacement for B1/B3, preserving the supplied source.
- The same shared-projection local/Mamba gated residual block for B2/B3, after
  `forward_features` and before the native classifier head.
- Native heads by default; the existing dense-head wrapper remains optional.
- Training-only inverse-frequency CE weights, with existing CE/focal alternatives.
- Whole-split metrics, ordinal five-class QWK, strict-best validation selection,
  early stopping, full optimizer/scheduler/scaler/RNG checkpointing and prediction exports.
- Publisher-provided splits and the explicitly deferred comprehensive leakage audit.

The CLI provides the proposed AdamW/warm-up/cosine recipe. The existing direct
`Trainer` API retains optimizer/scheduler/loss customization for planned extensions.
No custom SSM, extra convolution branch, bidirectional scan, new loss objective,
pretraining policy or data-split methodology was introduced.

## Runtime reporting and saved-weight exports — 17 October 2026

Training and checkpoint evaluation now print device/GPU, model and parameter
details, image size, class order, split sizes, per-class counts and loader settings.
Label counts use metadata, including nested Subsets, without decoding images.
`verbose=False` suppresses these messages; `evaluate.py --quiet` retains JSON-only
CLI output.

Completed fits export full best/last checkpoint copies to `saved_weights` by
default, with backbone, variant/method, run name, seed and epoch in each filename.
`training.saved_weights_dir` changes the destination or disables copies with null.
The copies retain reconstruction/preprocessing metadata and work with `evaluate.py`.
Export happens before automatic testing, including after early stopping.

Focused CPU verification: **7 runtime/export checks and 6 existing training-progress
regressions passed**. Checks covered distinct best/final weights, checkpoint reload,
metadata-only split counts, quiet output, undefined-QWK fallback, early stopping,
resume, and saving before a simulated test failure. A synthetic five-class B0 run
was evaluated through both the Python and CLI interfaces from its exported file,
after moving the original train/validation directories out of the way. Syntax and
CLI-help checks passed. Ruff was unavailable in the local environment. No full
dataset training or CUDA verification was performed for this update.

## Current verification

**Environment:** Python 3.12.14, PyTorch 2.5.1+cpu, torchvision 0.20.1+cpu,
timm 0.9.16, NumPy 1.26.4, scikit-learn 1.5.2. No CUDA device is available.
The supplied GPU environment was retained; it was not installed or certified here.

```bash
FDMAMBDR_SKIP_WORKERS=1 python -m unittest discover -s tests -v
python -m ruff check .
python -m ruff format --check .
python -m compileall -q .
python train.py --help
python evaluate.py --help
```

**66 tests discovered: 58 passed, 8 skipped, 0 failures/errors.** Seven skips
require CUDA (real Mamba, AMP or CUDA tensor tests). One explicit worker-process
skip is necessary because this execution environment denies worker IPC sockets;
an initial worker attempt failed at that environment restriction. The normal
local test command leaves the worker test enabled. Full output is retained in
`TEST_RESULTS.txt`. Lint, formatting, compilation and CLI help checks passed.

Verified checks include:

- B0/B1 for all three backbones at 224-pixel input, singleton batches, finite
  losses/gradients, optimizer updates and evaluation.
- Active FDConv spectral/KSM/FBM gradients, odd FFT grids, single-kernel mode,
  spectral/spatial parameterization agreement, state serialization and CPU autocast.
- Factory/direct-constructor equality for seeded MobileNet B0/B1, and rebuilding
  those models from the saved factory configuration.
- CE/focal formula and gradient comparisons; weighted loss/gradient accumulation
  consistency with incomplete batches; metric comparisons against scikit-learn.
- Correct best-vs-last selection, undefined QWK handling, no test-batch access
  during fitting, state restoration after evaluation errors and exact tested
  epoch-boundary resume with all process RNGs and `num_workers=0`.
- A synthetic five-grade ImageFolder run through the B1 training entry point,
  checkpoint export, then frozen external evaluation after removing training and
  validation files. Every external example was exported. This verifies plumbing,
  **not DR accuracy or generalization**.
- Fully frozen evaluation, pretraining/unused-option rejection, normalization
  mismatch rejection, deterministic-mode reset and missing-best resume rejection.

## Remaining boundaries and deferred work

1. **CUDA:** Real B2/B3 execution and GPU AMP are unverified. Run the full suite
   on the intended GPU environment without CUDA skips before training. No fake
   Mamba module was used to claim a pass.
2. **Actual data:** No dataset was attached. Labels, corrupt-file prevalence,
   patient independence and duplicate families remain unverified. Folder loaders
   validate class names, nonempty partitions and root separation; decoding errors
   propagate. Fingerprints hash ordered paths/labels, **not image bytes**. They do
   not establish leakage freedom or detect identical images under different paths.
3. **Data adapters and tooling:** CSV manifest generation/loaders, exact-file and
   near-duplicate audits, priority-eight/full-twelve launchers, benchmarks and
   reporting plots are still planned extensions. This cleanup does not claim the
   entire research execution plan is complete.
4. **Backbone comparability:** MobileNet/EfficientNet use 1x1 pointwise FDConv
   without default FBM; ResNet uses 3x3 with FBM. See `ARCHITECTURE.md` for sites.
5. **Resume:** Exact replay is tested at epoch boundaries with zero workers.
   Persistent worker augmentation RNGs are not checkpointed. Full old trainer
   checkpoints may fail the stricter/new model-configuration signature; see the
   compatibility section in `ARCHITECTURE.md`.
6. **Precision/options:** Keep model weights FP32 and use autocast. Direct
   `model.half()`, torch.compile, arbitrary FDConv nondefault parameters and
   unsupported input sizes have not been certified. Native BatchNorm retains its
   usual batch/spatial-size constraints.
7. **Evaluation:** The CLI reconstructs native-head factory models. Custom heads
   or custom datasets use the Python API and an explicitly matching constructor.
   Evaluation requires known deterministic transforms and saved normalization;
   custom transform behavior remains the caller's responsibility.
8. **Claims:** No full training, real DR prediction scores, clinical validity,
   performance advantage, latency, FLOPs or external independence is reported.

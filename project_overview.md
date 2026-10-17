# FD-MambaDR — Research Specification and Project Overview

> **Canonical handoff document for ChatGPT, Codex, and future contributors.** Read this file before modifying the study design or implementation. It records the latest decisions as of **17 October 2026** and distinguishes committed requirements, provisional choices, and deferred work. The project is an **undergraduate final-year project** and a scoped precursor to a later, broader frequency-adaptive state-space learning research program.
>
> **Status:** Architecture and evaluation methodology are proposed. No dataset audit, completed GPU training, accuracy improvement, external generalization, or clinical deployment has been verified in this document. Prior code bundles are starting points, **not evidence of passing end-to-end CUDA tests**.

## 0. Read this first — decisions and non-decisions

### Fixed research decisions

1. **Title:** *FD-MambaDR: Frequency-Dynamic Convolution and State-Space Feature Modeling for Efficient Diabetic Retinopathy Grading*.
2. **Task:** Supervised **five-class** diabetic retinopathy (DR) severity classification from color fundus images; labels `0,1,2,3,4` (No DR, Mild, Moderate, Severe, Proliferative DR), subject to verified source-label mapping.
3. **Backbone architectures:** `MobileNetV3-Large`, `EfficientNet-B0`, `ResNet-50`.
4. **Absolutely no pretrained model weights in primary experiments.** Create backbone architectures with random initialization; train *every* baseline and hybrid from scratch. This applies to CNN, FDConv, Mamba, projection, fusion, and head parameters. **Do not use** the pretrained skin-segmentation checkpoints bundled in MambaLiteUNet. A pretrained embedding model can be used **only for optional data auditing**, clearly separated from the training pipeline.
5. **Variants:** `B0` baseline, `B1` + selective FDConv, `B2` + lightweight Mamba fusion, `B3` + FDConv and Mamba. Three backbones × four variants = **12 primary configurations**.
6. **Training/development dataset:** Kaggle combined **EyePACS + APTOS 2019 + Gaussian-filtered APTOS + original Messidor** collection, described as 92,501 images initially and 143,669 image files after offline augmentation, resized to 600 × 600. These are *publisher-reported file counts*, not verified counts of independent patient retinas.
7. **Immediate workflow decision:** **Defer the comprehensive duplicate, provenance, and patient-level leakage audit** to save time before the final-year presentation. Use the dataset-provided 80/10/10 splits for **preliminary** training/validation/internal testing, where available. Do not present them as demonstrably leakage-free. Do minimal integrity/label checks, and preferably an exact-file cross-split duplicate scan, before training. Rebuild leakage-safe partitions later.
8. **External evaluation:** Prioritize **IDRiD** and **DDR** as separate datasets; **BRSET** is optional and **Messidor-2** is conditional upon overlap and five-grade label verification. Evaluate fixed checkpoints *without external training, tuning, or threshold optimization*; independence remains provisional until overlap checks are done.
9. **Primary selection/reporting metric:** Validation **quadratic weighted kappa (QWK)** for best checkpoint; also macro-F1, balanced accuracy, confusion matrix and relevant computational metrics.
10. **Reproducibility:** Every run must save configuration, training history, `best.pt`, `last.pt`, label mapping, seed, data-manifest identity, predictions, metrics, and environment/hardware details. No invented results.

### Provisional design choices (adjust only with documentation)

- Initial input size `224 × 224`; optional `256 × 256` or higher-resolution experiment, recognizing that tiny lesions may be lost at 224.
- Start with **one FDConv replacement** in a compatible intermediate/late convolutional location per backbone; no blanket replacement.
- Start with **one** lightweight local–global Mamba fusion block at a reduced spatial feature map (typically at stride 16 or 32), with bottleneck width and exact insertion point selected by compatibility checks and **validation-only** experiments.
- Start with AdamW, class-weighted cross-entropy, warm-up plus cosine decay, 80–150 *maximum planned* epochs with early stopping. Actual training length, batch size and hyperparameters depend on hardware and validation behavior.

### Explicitly deferred or out of scope for the final-year deadline

- Full near-duplicate/image-family/patient grouping and reconstructed group-safe partitions; retrospective audit required before stronger scientific claims.
- Formal clinical validation, deployment/diagnostic claims, lesion segmentation, an entirely new state-space algorithm, exhaustive SOTA benchmarking, and full-scale long-term frequency-adaptive SSM research.
- Mandatory training of all 12 full-length runs if compute limits intervene; the **8-configuration priority suite** is an acceptable scoped milestone, provided omitted experiments are clearly identified.

---

## 1. Problem, motivation, and scientific objectives

Retinal fundus images vary in illumination, pigmentation, camera quality and visible lesion characteristics. DR severity depends on small, local abnormalities (e.g., microaneurysms and exudates) as well as their extent and relationships across retinal regions. CNNs provide effective local representations but normally use input-independent learned convolution kernels; frequency-dynamic filtering offers content-adaptive kernel construction, while state-space sequence processing offers a route to context integration without reproducing a full Vision Transformer.

**Central research question:** *Do selective frequency-dynamic convolution and lightweight Mamba-based local–global feature fusion provide complementary, reproducible gains in five-class DR grading, and are those gains worth the parameter, memory and latency overhead?*

**Goals**

- G1 — Build three architecture-matched **scratch-trained** CNN baselines with reproducible data processing and checkpoints.
- G2 — Integrate the **actual provided FDConv design** into a small number of suitable `Conv2d` operations, preserving tensor semantics and gradients.
- G3 — Adapt lightweight Mamba and gated local–global feature mixing from the **provided MambaLiteUNet concepts** to *image classification*, without transplanting its segmentation decoder.
- G4 — Complete controlled B0/B1/B2/B3 ablations; test additivity/interaction rather than attributing any hybrid gain to one component without evidence.
- G5 — Evaluate internal grade discrimination and **zero-adaptation external** generalization on independently sourced datasets, documenting unverified overlap.
- G6 — Characterize accuracy–efficiency tradeoffs and deliver checkpoints, plots, experiment manifests and a transparent final-year research report.

**Research questions:** RQ1 (FDConv effect); RQ2 (Mamba effect); RQ3 (interaction of FDConv and Mamba); RQ4 (cross-backbone generality); RQ5 (cross-source generalization); RQ6 (computational overhead). All admit null or adverse findings.

**Expected contribution** is a carefully tested **integration and experimental study** for DR grading, *not* invention of FDConv, Mamba, or any claimed new SOTA algorithm. Never promise superiority over ViT-based models before measured evidence.

## 2. Technical architecture: FD-MambaDR

### 2.1 End-to-end schematic

```text
RGB fundus photograph [B,3,H,W]
    |
    +-- Image decoding, safe FOV crop, deterministic resize and normalization
    |   +-- TRAIN only: mild geometrical/photometric augmentation
    |
    v
CNN feature hierarchy (scratch initialized)
    MobileNetV3-Large / EfficientNet-B0 / ResNet-50
    |
    +-- For variants B1 and B3 ONLY:
    |      replace exactly one validated compatible spatial Conv2d
    |      with FDConv (FDW + KSM, FBM when enabled and compatible)
    |
    v
Selected spatial feature F [B,C,h,w], from defined stage
    |
    +--------------------+----------------------+
    |                    |                      |
    |             LOCAL PATH               CONTEXT PATH
    |             P_l: 1x1 Conv            P_m: 1x1 Conv
    |             [B,d,h,w]                [B,d,h,w]
    |             DWConv 3x3               flatten grid [B,L,d], L=h*w
    |             PWConv / activation       LayerNorm -> Mamba
    |             D [B,d,h,w]              reshape -> M [B,d,h,w]
    |                    |                      |
    +--------------------+----------------------+
                         |
          sigmoid gate from concat(D,M), size [B,d,h,w]
                         |
                  Q = G*M + (1-G)*D
                         |
               1x1 projection d -> C
                         |
           residual F_out = F + gamma*P_out(Q)
                         |
            remaining CNN stages if insertion is intermediate
                         |
                global average pooling -> dropout
                         |
              linear classification head -> 5 logits
                         |
           softmax probabilities / argmax DR grade 0..4
```

The figure is a **proposed data flow**. FDConv is a *replacement inside the backbone*, not necessarily an additional serialized block after a completed CNN. B0/B1 bypass the fusion block. B0/B2 retain standard convolutions at the selected FDConv sites. The network head must have comparable capacity across paired experiments except for intentionally added modules.

### 2.2 Baselines: exact architecture families

- **MobileNetV3-Large:** Inverted residual blocks, depthwise convolutions, squeeze/excitation and hardware-oriented layers. Treat it as the primary lightweight ablation backbone.
- **EfficientNet-B0:** MBConv blocks, depthwise operations and squeeze/excitation. A separate moderate-complexity architecture comparison.
- **ResNet-50:** Bottleneck residual stages with accessible `3×3`, `groups=1` spatial convolutions; good first target for testing source FDConv compatibility.
- Build from `timm` or `torchvision` with explicit **`pretrained=False` / `weights=None`** and `num_classes=5`; assert that no download or checkpoint load occurs. Store library/version and resolved module names. Do not assume `timm` and `torchvision` expose identical names or layer layouts.
- Randomly initialize *all* model variants independently under a documented seed and matching initialization policy. For fair B0/B1/B2/B3 comparisons, optionally copy the same **randomly initialized shared-backbone state** before changing components; this is NOT pretrained transfer. Record if paired initializations are used. Avoid inadvertent loading of an already trained baseline into a hybrid.

### 2.3 FDConv insertion contract

**Source:** supplied `FDConv.py`, based on Chen et al. It includes `FDConv(nn.Conv2d)`, `KernelSpatialModulation_Global`, `KernelSpatialModulation_Local`, `FrequencyBandModulation`, Fourier-domain kernel construction and `mmcv.cnn.CONV_LAYERS` registration.

The implementation uses:

1. **FDW — Fourier Disjoint Weight:** spectral coefficients, frequency assignments, inverse FFT kernel reconstruction.
2. **KSM — Kernel Spatial Modulation:** global and optional local content-conditioned modulation of convolution kernels.
3. **FBM — Frequency Band Modulation:** optional frequency-split feature modulation before the effective convolution.

Module abstraction (not a substitute for source implementation):

\[
\widetilde F=\operatorname{FBM}(F),\qquad
Y=\operatorname{Conv}\big(\widetilde F;W_{\mathrm{FD}}(F)\big).
\]

**Required engineering constraints**

- Enumerate all potential convolution replacements and print module path, `in_channels`, `out_channels`, kernel, padding, dilation, stride, groups, spatial feature size, original/replacement parameters.
- **Do not blindly replace depthwise/grouped convs.** The provided source forms Fourier weights using channel dimensions and has branches that may not be valid for every `groups != 1` configuration. Begin with a valid `groups=1`, `3×3` eligible ResNet bottleneck and test other families explicitly. Avoid silently substituting an unrelated dynamic-convolution module while calling it FDConv.
- **Eligibility/configuration:** the source defaults include channel thresholds and eligible kernel/stride lists. An FDConv object may fall back to ordinary `nn.Conv2d` when conditions do not match; assert that replacement is **active** (e.g., the spectral parameter/KSM modules exist) and that the intended module is invoked.
- Preserve output tensor shape, stride/padding/dilation and expected bias behavior. Check gradients to spectral parameters, KSM, FBM and upstream layers; finite loss and gradients after one optimizer step.
- Inspect the imported `mmcv` registration dependency. If decoupling it, preserve the numerical algorithm; document every source modification. Pin compatible PyTorch and optional MMCV versions when relevant.
- The source `profile_module` references undefined attributes and is not a trustworthy standalone FLOP estimator. Use runtime profiling and explain unsupported FFT/dynamic-kernel accounting.
- Numerical note: FFT components may cast to float32 even during AMP. Verify mixed-precision behavior; never assume `torch.compile`, half precision or FLOP profilers work without validation.
- **Scratch initialization:** conversion of an initial random spatial kernel to Fourier coefficients is allowed; conversion of *pretrained weights* is prohibited. Verify `convert_param` behavior and trainability after replacement.
- Begin with **one replaced layer** per backbone; additional layer count/location ablations are optional. Freeze the chosen location in run configs before final test-set evaluation.

**Implementation note (17 October 2026):** The current adapter uses an ungrouped
`1×1` pointwise convolution in MobileNetV3/EfficientNet-B0 because their candidate
spatial convolutions are depthwise, and an ungrouped `3×3` convolution in ResNet-50.
FDW/KSM are active in each; source-default FBM is enabled only at the `3×3` site.
Record this cross-backbone difference and the resolved module paths; see
`docs/IMPLEMENTATION_STATUS.md` for the tested timm version and exact locations.

### 2.4 Mamba local–global fusion contract

**Source inspiration:** supplied `MambaLiteUNet.zip`, containing its model, config, training scripts and pretrained segmentation weights. The underlying paper concerns skin lesion **segmentation**, so the U-Net-style decoder and skin-segmentation head are **not** transferred into this classification project. The classification fusion block is **our adaptation inspired by**, not an exact MambaLiteUNet clone.

For features \(F\in\mathbb R^{B\times C\times h\times w}\), let reduced width \(d\) be configurable (e.g., `d=32 or 64` as hardware-dependent *starting candidates*, not fixed validated settings). With \(L=hw\):

\[
Z=\operatorname{Flatten}_{hw}(P_m(F))\in\mathbb R^{B\times L\times d},\quad
M=\operatorname{Reshape}_{h,w}\big(\operatorname{Mamba}(\operatorname{LN}(Z))\big).
\]
\[
D=\operatorname{PWConv}(\operatorname{DWConv}_{3\times3}(P_l(F))),\quad
G=\sigma(\operatorname{Conv}_{1\times1}([D,M])),
\]
\[
F_{out}=F+\gamma P_{out}\big(G\odot M+(1-G)\odot D\big),
\]
where `gamma` is a learned residual-scale parameter initialized to a small value, and spatial dimensions remain unchanged.

**Required checks:** `B,C,h,w` input/output equality; order of raster scanning; gradient flow through both branches and gate; Mamba selective-scan CUDA availability; dimensions divisible/compatible with its requested internal expansion; test with `B=1`; ensure no segmentation masks or decoder paths are assumed. A fallback generic convolution is **not** equivalent to a true Mamba variant. If Mamba cannot run in the available environment, mark B2/B3 blocked and report why.

**Mamba placement:** favor spatially reduced feature maps because sequence length is \(h\times w\). Record exactly whether fusion runs *before final backbone stages* or *before global pooling* for each backbone. Preserve comparable feature semantics as far as architectures allow.

**Implementation note (17 October 2026):** Both B2 and B3 currently use the same
fusion class after `forward_features` and before the original classification
head. The implementation shares one input bottleneck projection between local
and Mamba branches, rather than the separate projections in the schematic. B3
adds only the selected backbone FDConv relative to B2; there is no additional
FDConv inside fusion. Real Mamba CUDA execution remains unverified locally.

### 2.5 Classification, initialization and experimental controls

- Output logits shape: `[B,5]`; softmax only for reported probabilities/inference, not before cross-entropy.
- Labels must be integer `torch.long`, `0..4`; no one-hot loss unless explicitly chosen.
- First loss: class-weighted cross-entropy with weights computed from **training labels only**. Apply the same weight policy to paired models.
- Use validation QWK for `best.pt` selection, with a deterministic tie policy; no test-informed early stopping.
- Avoid unreported architecture changes. If batch norm or dropout settings differ intentionally, save them in config and explain them.
- Parameter-efficient is an **empirical comparison**: record parameters/MACs/latency and never claim a guaranteed small overhead or superiority to all ViTs.

## 3. Datasets, labels and current short-term data decision

### 3.1 Main dataset: combined Kaggle repository

User-provided description names **“Eyepacs, Aptos, Messidor Diabetic Retinopathy”**, combining `EyePACS`, `APTOS 2019`, `APTOS (Gaussian Filtered)`, and original `Messidor` images. Publisher-reported statistics:

| Property | Reported value / interpretation |
|---|---|
| Before further offline augmentation | 92,501 JPG files |
| After 5 Jan 2024 manual augmentation | 143,669 JPG files (~55% increase) |
| Resized size | 600 × 600 pixels |
| Published partitions | 80% train / 10% validation / 10% test |
| Labels | Assumed 5-class grading **pending actual CSV/folder inspection** |
| Independent images/patients | **Unknown**; derived variants inflate file counts |
| Exact Kaggle dataset download URL | **Not yet recorded**; verify the precise version on disk |

**Immediate operational path, by user decision:** Use the provided partition folders/CSVs, **do not block initial training on a months-long leakage audit**. Preserve the original split manifest and dataset-version metadata. At minimum inspect readable files, grade ranges/counts and path existence; ideally SHA-256 cross-split exact duplicates. Do not reshuffle casually after model training begins.

**Language for reports:** Call results *“preliminary evaluation under the dataset-provided splits; augmentation-family and patient overlap not yet fully audited.”* Do not say *“independent internal test”*, *“no leakage”*, or *“patient-independent”* without evidence. If later audit detects significant leakage, withdraw/recompute affected claims and rerun the experiments with corrected splits.

### 3.2 Required basic loader behavior

- Support dataset-folder labels or metadata CSVs; **do not infer a specific unpublished folder structure**. Implement a manifest generator that verifies the actual downloaded layout and emits standardized records.
- Minimal internal manifest: `image_path,label,split`; recommended: `sample_id,image_path,label,split,source_dataset,original_id,group_id,is_augmented` when verifiable. Unknown metadata should be null/blank—not fabricated.
- Validate file count, missing paths, supported image types, five-class label mapping, class distribution, non-overlapping **file paths** and known SHA-256 duplicate collisions across partitions. Keep original filename/path information for traceability.
- Read RGB with PIL or equivalent; optionally crop black borders without erasing lesions; resize to `224×224` initially; train-only mild rotations/flips/brightness-contrast variation. Validation and test transforms must be deterministic. Avoid double-strength augmentation of pre-augmented training files; record whether provided offline augmentation is used.
- **Scratch training ≠ raw pixel normalization.** Input normalization (training-set mean/std) is permitted and must be fitted on train only; it is not weight pretraining. Test datasets use the fixed training normalization, not statistics fitted on external sets.
- Internal validation is used for hyperparameters/early stopping. Internal test is used only after a run's architecture and checkpoint-selection policy are frozen.

### 3.3 Later full audit (deferred, not forgotten)

The dataset merges raw APTOS and Gaussian-filtered APTOS and may contain offline image augmentations. Future full audit:

1. Inventory image names, source provenance, IDs, labels, dimensions and offline transforms.
2. Group same-origin images, both eyes of known patients and augmentations; detect exact duplicates using SHA-256 and near-duplicates using pHash/feature retrieval plus human verification.
3. Review conflicts and excluded/ungradable samples; split unique **groups** rather than filenames; apply augmentation only within training.
4. Check overlaps between internal partitions and between the whole development collection and external datasets.
5. Regenerate immutable manifests, record audit methods and rerun key results before making stronger generalization claims.

No similarity threshold is a guarantee of independence. In particular, a high embedding score alone must not trigger automatic deletion of a retinal image.

### 3.4 External evaluation

| Dataset | Planned role | Restrictions |
|---|---|---|
| **IDRiD** | Primary external evaluation, five-grade DR annotations | Do not train/tune on its labels in the zero-adaptation protocol; confirm available samples and grade mapping |
| **DDR** | Secondary larger external evaluation | Verify the chosen release, permission and `0..4` mapping; check overlap when feasible |
| **BRSET** | Optional additional geographic/site diversity | Access/metadata/label compatibility; patient-level and license constraints |
| **Messidor-2** | Conditional alternative | Separate from original Messidor, but must check image overlap and verified five-grade adjudicated annotations |

**Strict zero-adaptation workflow:** Train on combined Kaggle training split; choose checkpoint by combined Kaggle validation only; evaluate once on combined internal test and independently on each external test set. Keep external metrics reported **per source**. No labeled external tuning, no retraining, no externally fitted normalization and no choosing the best result after looking at external scores. If hyperparameters are revised after exposure to external labels, those datasets become development data; a new untouched external dataset is then needed for independent confirmation.

**Caveat:** External-source names alone do not prove no duplication with the combined repository; until full overlap audit, describe generalization conclusions as preliminary.

## 4. Complete experimental matrix

Each entry below is an independent run with its own checkpoint directory; all primary variants receive the same training data, augmentation policy, selection rule and comparable compute budget.

| Run ID | Backbone | FDConv | Mamba | Priority |
|---|---|---:|---:|---|
| `MNV3_B0` | MobileNetV3-Large | No | No | Core |
| `MNV3_B1` | MobileNetV3-Large | Yes | No | Core |
| `MNV3_B2` | MobileNetV3-Large | No | Yes | Core |
| `MNV3_B3` | MobileNetV3-Large | Yes | Yes | Core |
| `EFFB0_B0` | EfficientNet-B0 | No | No | Core |
| `EFFB0_B1` | EfficientNet-B0 | Yes | No | Extension |
| `EFFB0_B2` | EfficientNet-B0 | No | Yes | Extension |
| `EFFB0_B3` | EfficientNet-B0 | Yes | Yes | Core |
| `RN50_B0` | ResNet-50 | No | No | Core |
| `RN50_B1` | ResNet-50 | Yes | No | Extension |
| `RN50_B2` | ResNet-50 | No | Yes | Extension |
| `RN50_B3` | ResNet-50 | Yes | Yes | Core |

**Eight-run minimum:** all four MobileNet variants plus `EFFB0_B0/B3` and `RN50_B0/B3`. **Full factorial:** all twelve. If backbone-specific constraints force FDConv into dissimilar feature stages, document that as a comparison limitation.

### 4.1 Primary experiments

| Code | Study | Essential outputs |
|---|---|---|
| `E00` | Data inventory/basic checks | Image counts, classes, dataset schema, known duplicate warnings |
| `E01` | Scratch-trained CNN baselines | Learning histories and saved best/last checkpoints per backbone |
| `E02` | FDConv-only ablation | `B1` vs `B0` on same backbone and split |
| `E03` | Mamba-only ablation | `B2` vs `B0` under identical training budget |
| `E04` | Hybrid FD-MambaDR | `B3` vs `B0/B1/B2`, interaction discussion |
| `E05` | Cross-backbone comparison | Metrics + parameter/latency differences |
| `E06` | Internal held-out testing | QWK, macro-F1, grade confusion and prediction CSVs |
| `E07` | Zero-adaptation IDRiD test | External metrics and label-count report |
| `E08` | Zero-adaptation DDR test | External metrics and label-count report |
| `E09` | Computational benchmarks | Parameters, model MB, latency, peak memory, profiled FLOPs caveats |
| `E10` | Results synthesis | Comparison tables, learning curves, ablation charts, uncertainty disclosure |

### 4.2 Secondary/optional experiments

- `A1` — FDConv location: early vs intermediate vs late **eligible** convolution.
- `A2` — Number of FDConv replacements: one vs two (only after successful compatibility tests).
- `A3` — Mamba stage/insertion point and reduced width (e.g., 32 vs 64).
- `A4` — Input resolution `224` vs `256` (and higher only if feasible).
- `A5` — Loss class weights vs unweighted baseline / sampling sensitivity.
- `A6` — Multiple random seeds (e.g., 42, 123, 2026), prioritized for the strongest matched baseline/hybrid pair; report mean ± SD if actually run.
- `A7` — Grad-CAM/error review; show illustrative behavior without treating saliency as proof of clinical reasoning.
- `A8` — Optional source-held-out training if trustworthy source IDs are later recovered (e.g., EyePACS+APTOS train, original Messidor test, with all related variants excluded from training).

For optional ablations, use validation to choose hyperparameters; **do not** choose among many settings by internal or external test score.

## 5. Training recipe (scratch training)

Initial suggestions to turn into run configs, **not validated optimal choices**:

| Component | Initial proposal |
|---|---|
| Weights | Random initialization; `pretrained=False` or `weights=None` everywhere |
| Resolution | `224×224` RGB; higher resolution sensitivity later |
| Optimizer | AdamW |
| Learning rate | Trial starting point `1e-3` for scratch learning; tune on validation only, especially with FDConv/Mamba |
| Weight decay | `1e-4` initial candidate |
| Scheduler | Short warm-up followed by cosine decay |
| Loss | Cross-entropy, optionally class weighted using training counts only |
| Epoch budget | Up to `80–150` planned; select/stop based on validation QWK and compute limits |
| Batch size | Hardware-dependent; keep effective batch and schedule comparable |
| Seeds | Fix Python/NumPy/PyTorch/CUDA seeds, log residual nondeterminism |
| AMP | Use when tested; FFT and Mamba kernel mixed-precision behavior needs checks |
| Selection | Highest validation QWK (document tie policy) |
| Saving | Save last every epoch and best on QWK improvement; resume restores optimizer/scheduler/scaler/RNG where possible |

**Important:** A 143k-file mixed repository and 12 scratch-trained architectures may be expensive. Perform a small subset/one-batch test first, train the eight priority runs, and scale to 12 as resources allow. Distinguish fast debugging experiments from full scientific runs. Do **not** implicitly switch to ImageNet weights when convergence is slow.

## 6. Measurements, analysis and reporting

### 6.1 Primary and secondary metrics

**Primary:** quadratic weighted kappa for K=5 ordered grades:

\[
\kappa_{QW}=1-\frac{\sum_{i,j} w_{ij} O_{ij}}{\sum_{i,j} w_{ij} E_{ij}},\qquad
w_{ij}=\frac{(i-j)^2}{(5-1)^2}.
\]

Store the observed count matrix and software implementation. Clarify that QWK is undefined for some degenerate distributions/outputs; make any failure explicit rather than silently substituting 0.

**Secondary classification:** macro-F1, balanced accuracy, raw accuracy, per-grade precision/recall/F1/support, confusion matrix and per-image predicted probabilities.

**Exploratory binary screen:** referable DR defined for this experiment as grade `>=2` against `0/1`, with sensitivity, specificity and ROC-AUC when both groups are represented. This simplified endpoint is *not* a comprehensive clinical referral criterion (e.g., it omits DME-specific indications).

**Efficiency:** exact parameter count, trainable parameters, serialized checkpoint size, throughput and batch-1 inference latency at declared device/resolution, peak VRAM, and FLOPs/MACs only where the profiler models actual operations. FDConv dynamic grouped conv and FFT kernels can cause profiler undercounting; report methodology and limitations.

### 6.2 Minimum reporting tables and visualizations

1. Main dataset grade distribution, original vs augmented counts if verifiable.
2. Architecture table: backbone, FDConv module path, Mamba insertion site/width, param increase and test conditions.
3. Baseline/hybrid internal QWK, macro-F1 and balanced accuracy.
4. Full MobileNet factorial ablation B0–B3 (and others if completed).
5. External IDRiD and DDR tables **separately**, with grade supports and overlap caveat.
6. Best-model vs baseline confusion matrices and per-grade error summaries.
7. Train/val loss and validation QWK histories, with selected best epoch.
8. QWK-vs-parameters and QWK-vs-latency comparison charts.
9. Optional qualitative retinal image/saliency examples with limitations.
10. Literature context table with dataset, split, pretraining, image size and metric distinctions; **never claim direct dominance** across unmatched protocols.

Report negative/null results faithfully. A hybrid can add parameters yet lose accuracy; those are meaningful research outcomes.

## 7. Software architecture and repository contract

Prefer **one configurable pipeline** rather than hard-coded training scripts for each architecture. Suggested layout (adapt to the actual checked-out repository; do not assume it already exists):

```text
FD-MambaDR/
├── AGENTS.md                      # Codex instructions: point here as canonical methodology
├── project_overview.md            # THIS FILE
├── README.md                      # setup and quick-start
├── environment.yml / requirements.txt
├── configs/
│   ├── dataset.yaml
│   ├── train_defaults.yaml
│   ├── models/                    # B0/B1/B2/B3 x 3 backbones
│   └── suites/                    # priority8 and full12
├── data/                           # data itself ignored by git
│   ├── raw/                        # Kaggle, IDRiD, DDR (external storage preferable)
│   └── manifests/                  # train, val, test, idrid, ddr; private paths configurable
├── src/
│   ├── data/                       # manifests, PIL loaders, transforms, checks
│   ├── models/
│   │   ├── backbones.py
│   │   ├── fdconv_adapter.py
│   │   ├── mamba_fusion.py
│   │   └── model_factory.py
│   ├── engine/                     # training, metrics, checkpoints, evaluation
│   └── utils/                      # seed, logging, environment, profiling
├── scripts/
│   ├── inspect_dataset.py
│   ├── prepare_manifests.py
│   ├── smoke_test.py
│   ├── train.py
│   ├── run_experiments.py
│   ├── evaluate.py
│   ├── evaluate_all.py
│   ├── benchmark.py
│   ├── summarize_results.py
│   └── generate_figures.py
├── tests/                          # unit + integration tests
├── experiments/                    # saved checkpoints and histories; ignored by git
├── results/                        # exported report material; ignored by git or versioned selectively
├── docs/
│   ├── RESEARCH_CONTEXT.md
│   ├── EXPERIMENT_PLAN.md
│   ├── ARCHITECTURE.md
│   ├── DATASET_PROTOCOL.md
│   └── IMPLEMENTATION_STATUS.md
└── references/                     # original FDConv.py and MambaLiteUNet source (license-aware)
```

### 7.1 Uniform model construction

```python
model = build_model(
    backbone="mobilenetv3_large",   # or efficientnet_b0 / resnet50
    variant="B3",                   # B0, B1, B2, B3
    num_classes=5,
    pretrained=False,               # must remain False for all primary runs
    fdconv_cfg={"target": "verified_module_path", "kernel_num": 4},
    mamba_cfg={"stage": "verified_stage", "d_model": 64},
)
```

This is an **interface target**, not a claim that this exact function signature is already implemented. Resolved target modules and parameter deltas must be logged automatically. Fail loudly when requested FDConv/Mamba was not actually inserted.

### 7.2 Data manifest contract

At minimum:

```csv
sample_id,image_path,label,split,source_dataset
```

For future audited protocols also allow:

```csv
original_id,patient_id,group_id,is_augmented,sha256,phash
```

Keep missing fields empty; avoid generating fake source or patient identifiers. Use portable paths resolved against a configured dataset root rather than hard-coded developer machine paths.

### 7.3 Checkpoint and result contract

For each `<run_id>_seed<seed>`:

```text
experiments/<run_id>_seed<seed>/
  config.yaml
  metadata.json                 # timestamp, commit hash if available, software/device details
  dataset_fingerprints.json     # split manifest hash, label map, subset counts
  best.pt                       # best validation QWK checkpoint
  last.pt                       # latest resumable checkpoint
  history.csv                   # train/val losses, QWK and other metrics, epoch, LR
  evaluations/
    internal_test_metrics.json
    internal_test_predictions.csv
    idrid_metrics.json
    idrid_predictions.csv
    ddr_metrics.json
    ddr_predictions.csv
  profiling/
    complexity.json
  figures/
```

Checkpoint state should include at least `model_state_dict`, `optimizer_state_dict`, `scheduler_state_dict`, `epoch`, `best_val_qwk`, `seed`, `model_config`, `class_names`, and scaler state if using AMP; ideally save RNG states and data-manifest hashes. `best.pt` must record the **updated** best score when saved. Do not overwrite best with latest at the end. Evaluation must be rerunnable **without retraining**.

In prediction CSVs, record sample ID, true label, predicted label and `p0..p4`; optional confidence and source; do not expose unnecessary personal identifiers.

### 7.4 Error-handling and logging requirements

- Fail on missing images, bad labels, empty validation partitions, unexpected logits shape or NaN loss.
- Validate that training samples are not simply being loaded from external-test roots.
- Make checkpoint resume deterministic where practical and prevent silent architecture mismatch on load.
- Protect evaluation: avoid writing test data into training caches/samplers; `model.eval()` and inference mode; no augmentation randomness.
- Save aggregate CSV/JSON tables from **actual model outputs**, not hand-entered literature numbers.
- Do not track datasets, checkpoint binaries, credentials or large intermediate tensors in Git. Keep `project_overview.md`, configs, scripts and manuals version-controlled.

## 8. Preflight test checklist and acceptance gates

Run these **before launching long GPU experiments**:

- [ ] Python files compile; import all core modules in the actual training environment.
- [ ] Dependency/CUDA check verifies GPU, driver, PyTorch, Mamba kernels and FFT support. No hidden downloads.
- [ ] Data loader reads one minibatch from each dataset; image `[B,3,H,W]`, integer labels `0..4`, valid paths.
- [ ] All three **B0** backbones output `[B,5]` and support backward training with random initialization.
- [ ] Every active **B1/B3** FDConv target path is printed; replacement actually activates spectral/modulation modules (not a fallback plain convolution).
- [ ] Group/depthwise and stride cases used are shape/gradient tested; any unsupported target is rejected before training.
- [ ] Every **B2/B3** Mamba block outputs the expected feature shape and receives finite gradients.
- [ ] B3 passes one CPU or GPU forward/backward optimizer step as appropriate to dependencies; use true CUDA tests for actual Mamba selective-scan code.
- [ ] Small training run logs loss and QWK; validation is deterministic; best vs last checkpoint semantics and resume work.
- [ ] Each dataset's class counts and provenance limitations are printed and saved.
- [ ] Cross-split exact duplicates, if checked, produce a warning/CSV, not an unsupported blanket “no leakage” assertion.
- [ ] External evaluation loads a frozen checkpoint, with no train-set operations, and exports metrics plus predictions.
- [ ] Benchmark script records hardware and discloses operator-profiling limitations.
- [ ] `run_experiments.py --dry-run` enumerates the **exact expected 8 or 12 IDs**.

**Verification status at handoff:** These are acceptance criteria, **not assertions that the tests already passed**. Prior packaged scripts require an audit against these requirements, particularly because earlier code may have assumed pretrained models or older APTOS-only splits.

## 9. Execution priority and short-term schedule

| Phase | Priority | Work item | Deliverable |
|---|---|---|---|
| P0 | Must | Inspect actual Kaggle layout/labels; create trustworthy *file-level* manifests for published splits | manifests, class distributions, basic integrity report |
| P1 | Must | Environment setup and model smoke tests, scratch initialization assertions | dependency report, forward/backward test logs |
| P2 | Must | Train MobileNetV3 B0–B3 | four retained checkpoints and histories |
| P3 | Must | Train EfficientNet-B0 and ResNet-50 B0/B3 | four more retained checkpoints |
| P4 | Must | Locked internal evaluation on saved checkpoints | QWK/F1/confusion matrices, prediction CSVs |
| P5 | High | External IDRiD and DDR evaluation on frozen checkpoints | source-specific metrics with caveats |
| P6 | High | Efficiency benchmarks, plots, failure/error analysis | report-ready figures and tables |
| P7 | Stretch | EFF/RN B1/B2, seeds, resolution/placement ablations | complete 12 matrix, uncertainty estimates |
| P8 | Later | Comprehensive family/patient overlap and full research-grade dataset audit | recomputed leakage-safe protocol and results |

The user originally planned roughly one development week before a presentation. Actual timelines depend on GPU resources; do not promise 12 scratch runs will fit. If compute becomes insufficient, report only genuinely completed experiments and retain all checkpoints for continuation.

## 10. How another ChatGPT chat or Codex should proceed

1. **Read this entire document** before making architectural, dataset, or training changes. Also inspect `AGENTS.md`, `docs/IMPLEMENTATION_STATUS.md` and actual repository code; this overview is the intended **methodological source of truth**, not evidence of what files are presently implemented.
2. **Confirm which source files are available**: supplied `FDConv.py`, supplied `MambaLiteUNet.zip`, current experiment suite, and actual Kaggle layout. Verify licenses/attributions before redistributing upstream code or weights.
3. **Do not silently reverse decisions**: especially no pretrained weights, no APTOS-only development assumption, no claim that provided splits are leakage-free, and no automatic use of original Messidor as an external set.
4. Before edits, summarize the proposed changes and their test criteria. Implement small, compatible changes, then run available tests and update `docs/IMPLEMENTATION_STATUS.md` with exact pass/fail/pending results.
5. Keep dataset-path and hardware specifics in config, not hard-coded in modules; ask for missing filesystem paths only when actually required to run.
6. Treat all experimental claims as provisional until measurable artifacts exist. Report **actual measured results** with checkpoint/run ID, seed, image count, evaluation split and hardware.
7. When methodology changes after this handoff, update this file and related instructions together; record date and reasoning. If a future user instruction conflicts, the **most recent explicit user decision wins**.

### Suggested initial Codex prompt

> Read `project_overview.md` fully, then `AGENTS.md`, all `docs/*.md` and the existing PyTorch code. This is a scratch-trained five-class DR grading project using MobileNetV3-Large, EfficientNet-B0, ResNet-50, selective original FDConv and adapted lightweight Mamba local–global fusion. The current dataset is the combined EyePACS/APTOS/Messidor Kaggle collection under its supplied 80/10/10 splits for **preliminary** experiments; the full leakage audit is deferred. IDRiD and DDR are proposed zero-adaptation external evaluations. Audit the implementation against the document, explicitly identify missing or unverified dependencies and tests, then implement the next highest-priority executable experiment while saving checkpoints and evaluation artifacts. Do not enable pretrained weights or invent outcomes. Update `docs/IMPLEMENTATION_STATUS.md` after each tested change.

## 11. Primary sources and attributions

**Foundational methods**

- L. Chen et al., “Frequency Dynamic Convolution for Dense Image Prediction,” CVPR 2025. [Paper](https://openaccess.thecvf.com/content/CVPR2025/html/Chen_Frequency_Dynamic_Convolution_for_Dense_Image_Prediction_CVPR_2025_paper.html). Source implementation provided separately as `FDConv.py`.
- A. Gu and T. Dao, “Mamba: Linear-Time Sequence Modeling with Selective State Spaces,” COLM 2024. [Paper](https://arxiv.org/abs/2312.00752).
- M. M. Rahman et al., “MambaLiteUNet: Cross-Gated Adaptive Feature Fusion for Robust Skin Lesion Segmentation,” CVPR 2026. [Paper](https://openaccess.thecvf.com/content/CVPR2026/html/Rahman_MambaLiteUNet_Cross-Gated_Adaptive_Feature_Fusion_for_Robust_Skin_Lesion_Segmentation_CVPR_2026_paper.html). Source supplied as `MambaLiteUNet.zip`; its segmentation weights must not enter this scratch-training study.
- K. He et al., “Deep Residual Learning for Image Recognition,” CVPR 2016.
- A. Howard et al., “Searching for MobileNetV3,” ICCV 2019.
- M. Tan and Q. V. Le, “EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks,” ICML 2019.

**Original and candidate dataset sources**

- [EyePACS / Diabetic Retinopathy Detection](https://www.kaggle.com/competitions/diabetic-retinopathy-detection)
- [APTOS 2019 Blindness Detection](https://www.kaggle.com/competitions/aptos2019-blindness-detection)
- [Gaussian-filtered APTOS example](https://www.kaggle.com/datasets/sovitrath/diabetic-retinopathy-224x224-gaussian-filtered)
- [Original Messidor](https://www.adcis.net/en/third-party/messidor/) — acknowledge Messidor program partners and comply with redistribution terms.
- [IDRiD](https://idrid.grand-challenge.org/) — P. Porwal et al., *Data* 2018, 3(3), 25.
- [DDR dataset](https://github.com/nkicsl/DDR-dataset)
- [BRSET](https://physionet.org/content/brazilian-ophthalmological/1.0.2/)
- [Messidor-2](https://www.adcis.net/en/third-party/messidor2/) — only after label/overlap verification.

**Dataset citation caution:** The exact URL and version of the combined 143,669-file Kaggle repository should be recorded when available. Do not cite the four underlying datasets as though they directly attest the quality or leak-freedom of the *aggregated, augmented* release.

---

**Latest recorded decision:** Comprehensive data auditing is deferred for the immediate final-year workflow; minimal integrity checks and transparent limitations remain. Primary models are **trained from scratch**, and the first target is a reproducible **eight-configuration** priority suite, expandable to 12.

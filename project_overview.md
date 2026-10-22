# FDMamba research scope

FDMamba studies frequency-dynamic convolution and lightweight Mamba feature fusion
for medical image classification. The current repository emphasizes four simple
operations: downloading data, training one configured model, image inference,
and benchmarking explicitly selected checkpoints.

The three backbone families are MobileNetV3-Large, EfficientNet-B0 and ResNet-50.
B0 is the native CNN, B1 adds one FDConv, B2 adds local/global Mamba fusion, and B3
combines both. All train from scratch. The model computation, FDConv sites and
native classifier paths are preserved; only class count changes per task.

Default datasets are PathMNIST, DermaMNIST, BloodMNIST and OrganAMNIST. Use any
three or all four, preserving official train/validation/test splits. Train a
separate classifier for each dataset. This studies a reusable architecture across
modalities, not a jointly trained or zero-shot universal medical model.

Primary comparisons pair B0–B3 within each backbone and dataset. Match resolution,
augmentation, batch size, accumulation, dropout and optimization settings. Use
validation for selection and hyperparameters; evaluate the selected checkpoint
on the official test split. The full design contains 48 configurations per seed,
but the scripts operate on individual runs and explicit checkpoint lists.

Select by validation macro-F1 by default. Report accuracy, MedMNIST-compatible
AUC, macro-F1, balanced accuracy, per-class results and confusion matrices. Record
native archive resolution separately from model input size: the initial recipe
is 28-pixel data resized to 224, not native MedMNIST-224. Store checkpoints,
configuration, training history and TensorBoard logs for every run.

Report multiple seeds and uncertainty only when actually measured. Parameter
counts are exact; evaluation throughput includes loading/preprocessing and is not
model-only latency. Official partitions and archive checks are not independent
patient/duplicate audits. No clinical validity or performance improvement is claimed.

Architecture details are in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md); commands
and data policy are in [README.md](README.md).

Sources: [MedMNIST](https://github.com/MedMNIST/MedMNIST),
[MedMNIST v2](https://arxiv.org/abs/2110.14795),
[Frequency Dynamic Convolution](https://openaccess.thecvf.com/content/CVPR2025/html/Chen_Frequency_Dynamic_Convolution_for_Dense_Image_Prediction_CVPR_2025_paper.html),
[Mamba](https://arxiv.org/abs/2312.00752).

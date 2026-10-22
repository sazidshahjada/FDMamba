# Data cache

Run `python download_data.py` from the repository root. MedMNIST downloads and
caches the four default datasets in `data/medmnist/`, reusing verified archives.
Use `--datasets`, `--root` and `--size` to choose a subset, destination or native
resolution. Training `data.root` and `data.size` must match the downloaded data.

Each archive contains the official train/validation/test splits. Files remain
here across experiments and are ignored by Git. The current loader delegates all
download/cache behavior to MedMNIST; old `.part` files are not used by the module.

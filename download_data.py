"""Download the four default datasets using MedMNIST's own cache and progress bars."""

import argparse

from tqdm.auto import tqdm

from utils.data import DEFAULT_DATASETS, dataset_info, load_dataset


def download_datasets(datasets=DEFAULT_DATASETS, *, root="data/medmnist", size=28):
    """Download each official archive once; it contains train, validation and test."""
    for name in datasets:
        dataset_info(name)
    for name in tqdm(datasets, desc="Datasets", unit="dataset"):
        # download=True lets MedMNIST verify/reuse cached files automatically.
        dataset = load_dataset(name, "train", root=root, size=size, download=True)
        tqdm.write(f"{name}: {dataset.info['n_samples']} | cached in {root}")
        del dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--root", default="data/medmnist")
    parser.add_argument("--size", type=int, choices=(28, 64, 128, 224), default=28)
    args = parser.parse_args()
    download_datasets(args.datasets, root=args.root, size=args.size)


if __name__ == "__main__":
    main()

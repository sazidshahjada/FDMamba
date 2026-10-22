"""Download MedMNIST archives with checksum validation and interrupted-transfer resume."""

import argparse
import hashlib
import http.client
import re
import urllib.error
import urllib.request
from pathlib import Path

from tqdm.auto import tqdm

from utils.data import DEFAULT_DATASETS, dataset_info, load_dataset


def _md5(path, chunk_size=1024 * 1024):
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_archive(name, *, root, size, retries=3):
    """Download one archive atomically, resuming a partial transfer when possible."""
    info = dataset_info(name)
    suffix = "" if size == 28 else f"_{size}"
    filename = f"{name}{suffix}.npz"
    destination = root / filename
    partial = root / f"{filename}.part"
    expected_md5 = info[f"MD5{suffix}"]
    url = info[f"url{suffix}"]

    if destination.is_file() and _md5(destination) == expected_md5:
        tqdm.write(f"Using downloaded and verified file: {destination}")
        return destination

    # Older MedMNIST versions download into the final path. Preserve such an
    # interrupted transfer as the resumable partial file.
    if destination.exists():
        if not partial.exists() or destination.stat().st_size > partial.stat().st_size:
            destination.replace(partial)
        else:
            destination.unlink()

    last_error = None
    for attempt in range(1, retries + 1):
        offset = partial.stat().st_size if partial.exists() else 0
        request = urllib.request.Request(url, headers={"User-Agent": "pytorch/vision"})
        if offset:
            request.add_header("Range", f"bytes={offset}-")

        try:
            with urllib.request.urlopen(request) as response:
                status = getattr(response, "status", response.getcode())
                content_range = response.headers.get("Content-Range")
                range_match = (
                    re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", content_range)
                    if content_range
                    else None
                )
                resumed = status == 206 and range_match and int(range_match.group(1)) == offset
                initial = offset if resumed else 0
                mode = "ab" if resumed else "wb"
                if range_match:
                    total = int(range_match.group(3))
                elif response.headers.get("Content-Length"):
                    total = initial + int(response.headers["Content-Length"])
                else:
                    total = None

                label = f"Resuming {filename}" if resumed else f"Downloading {filename}"
                with partial.open(mode) as stream, tqdm(
                    total=total,
                    initial=initial,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=label,
                ) as progress:
                    for chunk in iter(lambda: response.read(1024 * 1024), b""):
                        stream.write(chunk)
                        progress.update(len(chunk))

            if total is not None and partial.stat().st_size != total:
                raise OSError(
                    f"incomplete response ({partial.stat().st_size} of {total} bytes received)"
                )
            if _md5(partial) != expected_md5:
                # A complete file with the wrong digest cannot be repaired by
                # appending. Restart it on the next attempt.
                partial.unlink()
                raise OSError("download completed but the MD5 checksum did not match")
            partial.replace(destination)
            return destination
        except (OSError, urllib.error.URLError, http.client.HTTPException) as error:
            last_error = error
            if attempt < retries:
                kept = partial.stat().st_size if partial.exists() else 0
                tqdm.write(
                    f"{filename}: attempt {attempt}/{retries} failed ({error}); "
                    f"retrying from {kept} bytes"
                )

    raise RuntimeError(
        f"Failed to download {filename} after {retries} attempts. "
        f"The partial transfer is kept at {partial}; rerun this command to resume."
    ) from last_error


def download_datasets(datasets=DEFAULT_DATASETS, *, root="data/medmnist", size=28, retries=3):
    """Download each official archive once; reuse valid files and resume partial ones."""
    for name in datasets:
        dataset_info(name)
    if not isinstance(size, int) or isinstance(size, bool) or size not in (28, 64, 128, 224):
        raise ValueError("size must be an official MedMNIST resolution: 28, 64, 128 or 224")
    if retries < 1:
        raise ValueError("retries must be at least 1")
    root = Path(root).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    for name in tqdm(datasets, desc="Datasets", unit="dataset"):
        _download_archive(name, root=root, size=size, retries=retries)
        dataset = load_dataset(name, "train", root=root, size=size, download=False)
        tqdm.write(f"{name}: {dataset.info['n_samples']} | cached in {root}")
        del dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--root", default="data/medmnist")
    parser.add_argument("--size", type=int, choices=(28, 64, 128, 224), default=28)
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args()
    download_datasets(args.datasets, root=args.root, size=args.size, retries=args.retries)


if __name__ == "__main__":
    main()

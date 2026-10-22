"""Train one FDMamba model/dataset and export selected weights to saved_weight/."""

import argparse

from FDMamba import BACKBONES
from utils.config import load_config
from utils.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--dataset", required=True, help="MedMNIST flag, e.g. pathmnist")
    parser.add_argument("--name", help="Custom experiment name, e.g. MobileNet-v1")
    parser.add_argument("--backbone", choices=BACKBONES)
    parser.add_argument("--variant", choices=("B0", "B1", "B2", "B3"))
    parser.add_argument("--device", help="auto, cpu, cuda or cuda:0")
    parser.add_argument(
        "--download", action="store_true", help="Let MedMNIST download missing data"
    )
    parser.add_argument("--resume", action="store_true", help="Continue last.pt in this experiment")
    args = parser.parse_args()
    config = load_config(args.config)
    for section, key, value in (
        ("model", "backbone", args.backbone),
        ("model", "variant", args.variant),
        ("training", "device", args.device),
    ):
        if value is not None:
            config[section][key] = value
    Trainer(config, args.dataset, name=args.name, download=args.download).fit(resume=args.resume)


if __name__ == "__main__":
    main()

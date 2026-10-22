"""Predict the class of one image using a selected model checkpoint."""

import argparse
import json

from utils.evaluation import predict_image
from utils.runtime import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--model", help="Expected backbone, full model name or saved custom name")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", help="Optional prediction JSON path")
    args = parser.parse_args()
    result = predict_image(
        args.checkpoint, args.image, model=args.model, device=args.device, top_k=args.top_k
    )
    if args.output:
        write_json(args.output, result)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

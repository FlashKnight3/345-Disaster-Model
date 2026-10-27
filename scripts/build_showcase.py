"""Build the offline demo page from saved predictions."""

import argparse
from pathlib import Path

from disaster_model.showcase import build_showcase


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/manifest.jsonl"))
    parser.add_argument("--predictions", type=Path, default=Path("results/demo/predictions.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("results/demo/showcase.html"))
    args = parser.parse_args()
    build_showcase(args.manifest, args.predictions, args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()

"""Make a small class-balanced xBD sample for the midterm demo."""

import argparse
from pathlib import Path

from disaster_model.xbd import prepare_sample


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xbd-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/processed"))
    parser.add_argument("--per-class", type=int, default=3)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=345)
    args = parser.parse_args()
    manifest = prepare_sample(args.xbd_root, args.output, args.per_class, args.crop_size, seed=args.seed)
    print(f"Wrote {args.per_class * 4} buildings to {manifest}")
    print("Inspect every crop before running the model; discard unclear examples.")


if __name__ == "__main__":
    main()

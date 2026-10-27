"""Create a contact sheet to inspect all selected before/after crops."""

import argparse
from pathlib import Path

from PIL import Image, ImageDraw

from disaster_model.data import read_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/manifest.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/contact_sheet.jpg"))
    args = parser.parse_args()
    buildings = list(read_manifest(args.manifest))
    if not buildings:
        parser.error("Manifest is empty")

    tile = 256
    label_height = 44
    rows = (len(buildings) + 1) // 2
    sheet = Image.new("RGB", (4 * tile, rows * (tile + label_height)), "white")
    draw = ImageDraw.Draw(sheet)
    for index, building in enumerate(buildings):
        column, row = index % 2, index // 2
        x = column * 2 * tile
        y = row * (tile + label_height)
        for offset, image_path in enumerate((building.pre_image, building.post_image)):
            with Image.open(image_path) as source:
                sheet.paste(source.convert("RGB").resize((tile, tile)), (x + offset * tile, y))
        draw.text((x + 5, y + tile + 3), f"{building.building_id[:36]}  |  {building.true_label}", fill="black")
        draw.text((x + 5, y + tile + 20), "before                              after", fill="black")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.output, quality=90)
    print(f"Inspect all {len(buildings)} building pairs: {args.output}")


if __name__ == "__main__":
    main()

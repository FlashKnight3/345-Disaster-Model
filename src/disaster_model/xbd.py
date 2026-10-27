"""Select labeled xBD buildings and make matching before/after crops."""

import hashlib
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw

from .conditions import DAMAGE_LABELS


SUBTYPE_TO_LABEL = {
    "no-damage": "no damage",
    "minor-damage": "minor damage",
    "major-damage": "major damage",
    "destroyed": "destroyed",
}
NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
POINT = re.compile(rf"({NUMBER})\s+({NUMBER})")


@dataclass(frozen=True)
class Candidate:
    building_id: str
    pre_image: Path
    post_image: Path
    true_label: str
    pre_bounds: tuple[float, float, float, float]
    post_bounds: tuple[float, float, float, float]
    disaster_type: str | None


def polygon_bounds(wkt: str) -> tuple[float, float, float, float]:
    """Read pixel-coordinate bounds from an xBD POLYGON or MULTIPOLYGON WKT."""
    if not wkt.lstrip().upper().startswith(("POLYGON", "MULTIPOLYGON")):
        raise ValueError("Expected polygon WKT")
    points = [(float(x), float(y)) for x, y in POINT.findall(wkt)]
    if len(points) < 4:
        raise ValueError("Polygon WKT has too few points")
    xs, ys = zip(*points)
    return min(xs), min(ys), max(xs), max(ys)


def _candidates(root: Path, crop_size: int, padding: int):
    labels = sorted(root.rglob("*_post_disaster.json"))
    if not labels:
        raise ValueError(f"No xBD post-disaster labels found under {root}")
    for label_path in labels:
        tile = label_path.stem.removesuffix("_post_disaster")
        pre_label_path = label_path.with_name(f"{tile}_pre_disaster.json")
        if not pre_label_path.is_file():
            continue
        image_dir = label_path.parent.parent / "images"
        pre = image_dir / f"{tile}_pre_disaster.png"
        post = image_dir / f"{tile}_post_disaster.png"
        if not pre.is_file() or not post.is_file():
            continue
        with Image.open(pre) as pre_source, Image.open(post) as post_source:
            if pre_source.size != post_source.size:
                continue
            image_size = pre_source.size
        annotation = json.loads(label_path.read_text(encoding="utf-8"))
        pre_annotation = json.loads(pre_label_path.read_text(encoding="utf-8"))
        pre_features = {
            feature.get("properties", {}).get("uid"): feature
            for feature in pre_annotation.get("features", {}).get("xy", [])
        }
        metadata = annotation.get("metadata", {})
        disaster_type = metadata.get("disaster_type")
        for feature in annotation.get("features", {}).get("xy", []):
            label = SUBTYPE_TO_LABEL.get(feature.get("properties", {}).get("subtype"))
            uid = feature.get("properties", {}).get("uid")
            if label is None or uid is None:
                continue
            try:
                pre_bounds = polygon_bounds(pre_features[uid]["wkt"])
                post_bounds = polygon_bounds(feature["wkt"])
            except (KeyError, ValueError):
                continue
            if min(
                pre_bounds[2] - pre_bounds[0], pre_bounds[3] - pre_bounds[1],
                post_bounds[2] - post_bounds[0], post_bounds[3] - post_bounds[1],
            ) < 24:
                continue
            if (_crop_box(pre_bounds, image_size, crop_size, padding) is None or
                    _crop_box(post_bounds, image_size, crop_size, padding) is None):
                continue
            yield Candidate(
                building_id=f"{tile}:{uid}",
                pre_image=pre,
                post_image=post,
                true_label=label,
                pre_bounds=pre_bounds,
                post_bounds=post_bounds,
                disaster_type=disaster_type,
            )


def _crop_box(bounds, image_size, crop_size, padding):
    x0, y0, x1, y1 = bounds
    width, height = image_size
    if (x0 < 0 or y0 < 0 or x1 > width or y1 > height or
            x1 - x0 > crop_size - 2 * padding or
            y1 - y0 > crop_size - 2 * padding):
        return None
    if width < crop_size or height < crop_size:
        return None
    left = round((x0 + x1 - crop_size) / 2)
    top = round((y0 + y1 - crop_size) / 2)
    if left < 0 or top < 0 or left + crop_size > width or top + crop_size > height:
        return None
    return left, top, left + crop_size, top + crop_size


def prepare_sample(
    xbd_root: Path,
    output_dir: Path,
    per_class: int = 3,
    crop_size: int = 256,
    padding: int = 24,
    seed: int = 345,
) -> Path:
    """Create a deterministic, class-balanced demo manifest and image crops."""
    if per_class < 1 or crop_size < 64 or padding < 0 or 2 * padding >= crop_size:
        raise ValueError("Invalid per-class count, crop size, or padding")
    rng = random.Random(seed)
    selected: dict[str, list[Candidate]] = {label: [] for label in DAMAGE_LABELS}
    seen = {label: 0 for label in DAMAGE_LABELS}
    for candidate in _candidates(xbd_root, crop_size, padding):
        label = candidate.true_label
        seen[label] += 1
        bucket = selected[label]
        if len(bucket) < per_class:
            bucket.append(candidate)
        else:
            index = rng.randrange(seen[label])
            if index < per_class:
                bucket[index] = candidate

    missing = [label for label, bucket in selected.items() if len(bucket) < per_class]
    if missing:
        raise ValueError(f"Not enough labeled buildings for: {', '.join(missing)}")

    # Selection is deterministic, but reject unusable image pairs rather than
    # silently producing crops of different areas or dimensions.
    records = []
    crops = output_dir / "crops"
    crops.mkdir(parents=True, exist_ok=True)
    for label in DAMAGE_LABELS:
        for candidate in selected[label]:
            with Image.open(candidate.pre_image) as pre_source, Image.open(candidate.post_image) as post_source:
                if pre_source.size != post_source.size:
                    raise ValueError(f"Image dimensions differ: {candidate.building_id}")
                pre_box = _crop_box(candidate.pre_bounds, pre_source.size, crop_size, padding)
                post_box = _crop_box(candidate.post_bounds, post_source.size, crop_size, padding)
                if pre_box is None or post_box is None:
                    raise ValueError(f"Building does not fit the crop: {candidate.building_id}")
                identifier = hashlib.sha256(candidate.building_id.encode()).hexdigest()[:16]
                pre_name = f"{identifier}-pre.png"
                post_name = f"{identifier}-post.png"
                for source, name, bounds, box in (
                    (pre_source, pre_name, candidate.pre_bounds, pre_box),
                    (post_source, post_name, candidate.post_bounds, post_box),
                ):
                    crop = source.convert("RGB").crop(box)
                    target_box = tuple(round(value - offset) for value, offset in zip(bounds, (box[0], box[1], box[0], box[1])))
                    ImageDraw.Draw(crop).rectangle(target_box, outline="yellow", width=3)
                    crop.save(crops / name)
            records.append({
                "building_id": candidate.building_id,
                "pre_image": f"crops/{pre_name}",
                "post_image": f"crops/{post_name}",
                "true_label": candidate.true_label,
                "disaster_type": candidate.disaster_type,
                "split": "demo",
            })

    manifest = output_dir / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
    return manifest

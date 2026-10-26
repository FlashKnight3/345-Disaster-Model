"""Manifest contract for paired, building-level image crops."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from .conditions import DAMAGE_LABELS


@dataclass(frozen=True)
class Building:
    building_id: str
    pre_image: Path
    post_image: Path
    true_label: str
    split: str
    disaster_type: str | None = None


def read_manifest(path: str | Path) -> Iterator[Building]:
    """Read JSONL records; resolve crop paths relative to the manifest."""
    manifest = Path(path)
    with manifest.open(encoding="utf-8") as rows:
        for line_number, line in enumerate(rows, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            label = record["true_label"]
            if label not in DAMAGE_LABELS:
                raise ValueError(f"{manifest}:{line_number}: invalid label {label!r}")
            yield Building(
                building_id=record["building_id"],
                pre_image=manifest.parent / record["pre_image"],
                post_image=manifest.parent / record["post_image"],
                true_label=label,
                split=record["split"],
                disaster_type=record.get("disaster_type"),
            )

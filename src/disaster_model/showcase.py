"""Turn a completed run into one HTML file that works offline."""

import base64
import hashlib
import html
import json
from collections import Counter, defaultdict
from math import isclose, isfinite
from pathlib import Path

from .conditions import ANSWER_CHOICES
from .data import Building, read_manifest


STYLE = """
body { font: 16px/1.5 system-ui, sans-serif; color: #222; max-width: 980px; margin: auto; padding: 24px; }
h1, h2 { line-height: 1.2; }
header { border-bottom: 2px solid #222; padding-bottom: 20px; }
.summary { background: #f3f3f3; padding: 12px 16px; }
.note, .source { color: #555; font-size: 14px; }
.example { border-bottom: 1px solid #ccc; padding: 24px 0 32px; }
.example h2 { margin-bottom: 6px; }
.example p { margin: 6px 0; }
.images { display: flex; gap: 16px; flex-wrap: wrap; margin: 16px 0; }
figure { margin: 0; }
img { display: block; width: 256px; height: 256px; }
figcaption { margin-top: 4px; }
.table-wrap { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; min-width: 660px; }
th, td { border-bottom: 1px solid #ddd; padding: 8px; text-align: left; vertical-align: top; }
th { background: #f3f3f3; }
summary { cursor: pointer; }
.shares { color: #555; font-size: 13px; margin-top: 6px; white-space: nowrap; }
code { overflow-wrap: anywhere; }
"""


def _image_uri(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def _choice_details(row: dict) -> str:
    shares = row["choice_probabilities"]
    if (set(shares) != set(ANSWER_CHOICES)
            or not all(isfinite(value) and 0 <= value <= 1 for value in shares.values())
            or not isclose(sum(shares.values()), 1.0, rel_tol=0, abs_tol=1e-8)
            or row["predicted_choice"] != max(shares, key=shares.get)
            or not isclose(row["confidence"], shares[row["predicted_choice"]], rel_tol=0, abs_tol=1e-8)):
        raise ValueError(f"Invalid A/B/C/D probabilities for {row['building_id']}")

    choices = "<br>".join(
        f"{choice}: {label} — {shares[choice]:.12g}"
        for choice, label in ANSWER_CHOICES.items()
    )
    return f"<details><summary>{row['confidence']:.12g}</summary><div class='shares'>{choices}</div></details>"


def _result_row(row: dict) -> str:
    names = {
        "image_only": "Images only",
        "matching_text": "Matching claim",
        "conflicting_text": "Conflicting claim",
    }
    claim = html.escape(row["text_claim"]) if row["text_claim"] else "—"
    gap = "—" if row["conflict_distance"] is None else str(row["conflict_distance"])
    changed = "—" if row["prediction_changed"] is None else ("Yes" if row["prediction_changed"] else "No")
    answer = html.escape(f"{row['predicted_choice']}: {row['predicted_label']}")
    return (
        "<tr>"
        f"<td>{names[row['condition']]}</td><td>{claim}</td><td>{gap}</td>"
        f"<td>{answer}</td><td>{_choice_details(row)}</td><td>{changed}</td>"
        "</tr>"
    )


def _building_section(number: int, building: Building, rows: list[dict]) -> str:
    tile = building.building_id.split(":", 1)[0]
    event = tile.rsplit("_", 1)[0].replace("-", " ").title()
    before = _image_uri(building.pre_image)
    after = _image_uri(building.post_image)
    table_rows = "".join(_result_row(row) for row in rows)
    return f"""
<section class="example">
  <h2>{number:02d}. {html.escape(event)}</h2>
  <p>xBD label: <strong>{html.escape(building.true_label)}</strong> · tile {html.escape(tile)}</p>
  <details class="source"><summary>Building ID</summary><code>{html.escape(building.building_id)}</code></details>
  <div class="images">
    <figure><img src="{before}" alt="Before disaster"><figcaption>Before</figcaption></figure>
    <figure><img src="{after}" alt="After disaster"><figcaption>After</figcaption></figure>
  </div>
  <div class="table-wrap"><table>
    <thead><tr><th>Input</th><th>Claim</th><th>Gap</th><th>Model answer</th><th>Share</th><th>Changed?</th></tr></thead>
    <tbody>{table_rows}</tbody>
  </table></div>
</section>"""


def build_showcase(manifest: Path, predictions: Path, output: Path) -> None:
    buildings = list(read_manifest(manifest))
    run = json.loads((predictions.parent / "run.json").read_text(encoding="utf-8"))
    if not run.get("complete"):
        raise ValueError("Inference run is incomplete")
    if run["manifest_sha256"] != hashlib.sha256(manifest.read_bytes()).hexdigest():
        raise ValueError("Results were produced from a different manifest")

    rows_by_building = defaultdict(list)
    for line in predictions.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows_by_building[row["building_id"]].append(row)
    if set(rows_by_building) != {building.building_id for building in buildings}:
        raise ValueError("Results do not cover every manifest building")
    expected = {"image_only": 1, "matching_text": 1, "conflicting_text": 2}
    for building in buildings:
        if Counter(row["condition"] for row in rows_by_building[building.building_id]) != expected:
            raise ValueError(f"Expected four demo conditions for {building.building_id}")

    all_rows = [row for rows in rows_by_building.values() for row in rows]
    baselines = [row for row in all_rows if row["condition"] == "image_only"]
    conflicts = [row for row in all_rows if row["condition"] == "conflicting_text"]
    eligible = [row for row in conflicts if row["image_only_prediction"] != row["text_claim"]]
    correct = sum(row["predicted_label"] == row["true_label"] for row in baselines)
    followed = sum(row["changed_toward_text"] for row in eligible)
    sections = "".join(
        _building_section(index, building, rows_by_building[building.building_id])
        for index, building in enumerate(buildings, start=1)
    )

    page = f"""<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Building damage demo</title><style>{STYLE}</style></head>
<body><main>
<header>
  <h1>Building damage demo</h1>
  <p>COMP 345 · {html.escape(run['model_id'])} · {len(buildings)} xBD buildings</p>
  <p class="summary"><strong>{correct}/{len(baselines)}</strong> image-only answers matched xBD.<br>
  <strong>{followed}/{len(eligible)}</strong> eligible conflicting claims moved the answer to the claim.</p>
  <p class="note">Eligible means the claim differed from the image-only answer. Gap is the number of damage classes between the claim and the xBD label. Click a share to see all four A/B/C/D values; they sum to 1 for each row. These are relative choice scores, not probabilities of being correct. This is a small demo sample.</p>
  <details class="source"><summary>Run details</summary>Model revision: <code>{html.escape(run['model_revision'])}</code></details>
</header>
{sections}
</main></body></html>"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(page, encoding="utf-8")

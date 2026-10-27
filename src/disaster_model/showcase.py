"""Build a self-contained HTML report from actual model outputs."""

import base64
import hashlib
import html
import json
from collections import Counter, defaultdict
from math import isclose, isfinite
from pathlib import Path

from .conditions import ANSWER_CHOICES
from .data import read_manifest


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


def build_showcase(manifest: Path, predictions: Path, output: Path) -> None:
    buildings = list(read_manifest(manifest))
    run_path = predictions.parent / "run.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    if not run.get("complete"):
        raise ValueError("Inference run is incomplete")
    if run["manifest_sha256"] != hashlib.sha256(manifest.read_bytes()).hexdigest():
        raise ValueError("Results were produced from a different manifest")

    rows_by_building = defaultdict(list)
    for line in predictions.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows_by_building[row["building_id"]].append(row)
    if len(rows_by_building) != len(buildings):
        raise ValueError("Results do not cover every manifest building")
    for building in buildings:
        rows = rows_by_building[building.building_id]
        if len(rows) != 4 or Counter(row["condition"] for row in rows) != {
            "image_only": 1, "matching_text": 1, "conflicting_text": 2
        }:
            raise ValueError(f"Expected four demo conditions for {building.building_id}")

    all_rows = [row for rows in rows_by_building.values() for row in rows]
    baseline_rows = [row for row in all_rows if row["condition"] == "image_only"]
    conflict_rows = [row for row in all_rows if row["condition"] == "conflicting_text"]
    eligible = [row for row in conflict_rows if row["image_only_prediction"] != row["text_claim"]]
    flips = sum(bool(row["changed_toward_text"]) for row in eligible)
    correct = sum(row["predicted_label"] == row["true_label"] for row in baseline_rows)

    cards = []
    for building in buildings:
        rows = rows_by_building[building.building_id]
        detail_rows = []
        for row in rows:
            claim = row["text_claim"] or "none"
            distance = "—" if row["conflict_distance"] is None else str(row["conflict_distance"])
            changed = "—" if row["prediction_changed"] is None else ("yes" if row["prediction_changed"] else "no")
            detail_rows.append(
                "<tr>"
                f"<td>{html.escape(row['condition'].replace('_', ' '))}</td>"
                f"<td>{html.escape(claim)}</td><td>{distance}</td>"
                f"<td>{html.escape(row['predicted_choice'])}: {html.escape(row['predicted_label'])}</td>"
                f"<td>{_choice_details(row)}</td><td>{changed}</td>"
                "</tr>"
            )
        cards.append(
            "<article class='card'>"
            f"<h2>{html.escape(building.building_id)}</h2>"
            f"<p>True label: <strong>{html.escape(building.true_label)}</strong></p>"
            "<div class='images'>"
            f"<figure><img src='{_image_uri(building.pre_image)}' alt='Pre-disaster crop'><figcaption>Before</figcaption></figure>"
            f"<figure><img src='{_image_uri(building.post_image)}' alt='Post-disaster crop'><figcaption>After</figcaption></figure>"
            "</div><table><thead><tr><th>Condition</th><th>Claim</th><th>Distance</th>"
            "<th>Prediction</th><th>Winning share</th><th>Changed?</th></tr></thead><tbody>"
            + "".join(detail_rows) + "</tbody></table></article>"
        )

    content = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Disaster damage demo</title><style>
body{{font:16px system-ui,sans-serif;color:#17212b;background:#f4f6f8;margin:0}}
main{{max-width:1100px;margin:auto;padding:28px}} h1{{margin-bottom:4px}} .muted{{color:#586776}}
.summary{{display:flex;gap:16px;flex-wrap:wrap;margin:24px 0}} .stat,.card{{background:white;border-radius:12px;padding:18px;box-shadow:0 2px 10px #0001}}
.stat{{min-width:170px}} .stat strong{{display:block;font-size:28px}} .card{{margin:20px 0}}
.images{{display:flex;gap:12px;flex-wrap:wrap}} figure{{margin:0}} img{{width:256px;height:256px;object-fit:cover}}
figcaption{{text-align:center;color:#586776}} table{{border-collapse:collapse;width:100%;margin-top:18px}}
th,td{{padding:9px;text-align:left;border-bottom:1px solid #e2e6ea}} th{{background:#eef2f5}}
.shares{{white-space:nowrap;margin-top:6px;line-height:1.5;font-size:13px;color:#435466}}
@media(max-width:650px){{table{{font-size:12px}}th,td{{padding:5px}}img{{width:44vw;height:44vw}}}}
</style></head><body><main><h1>Multimodal disaster damage demo</h1>
<p class="muted">Pilot sample · model {html.escape(run['model_id'])} · revision {html.escape(run['model_revision'][:12])}</p>
<div class="summary"><div class="stat"><strong>{len(buildings)}</strong>buildings</div>
<div class="stat"><strong>{correct}/{len(baseline_rows)}</strong>image-only correct</div>
<div class="stat"><strong>{flips}/{len(eligible)}</strong>conflicting claims that drew a prediction toward text</div></div>
<p>Each row's A/B/C/D shares sum to 1. The winning share is shown; click it to see all four. These shares are relative to the four choices, not calibrated probabilities of correctness. This small pilot is descriptive only.</p>
{''.join(cards)}
</main></body></html>"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(content, encoding="utf-8")

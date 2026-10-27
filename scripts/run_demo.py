"""Run pinned Qwen3-VL on a prepared xBD demo manifest."""

import argparse
import hashlib
import json
from pathlib import Path

from disaster_model.data import read_manifest
from disaster_model.demo import demo_conditions
from disaster_model.qwen import QwenChoiceScorer
from disaster_model.results import result_row
from disaster_model.scoring import prediction_from_scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/manifest.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("results/demo"))
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--revision", help="Hugging Face commit SHA; current commit is resolved if omitted")
    parser.add_argument("--allow-cpu", action="store_true", help="Allow a slow CPU pilot")
    parser.add_argument("--pilot-only", action="store_true", help="Score only the first image-only example")
    args = parser.parse_args()

    buildings = list(read_manifest(args.manifest))
    if not buildings:
        parser.error("Manifest is empty")
    ids = [building.building_id for building in buildings]
    if len(ids) != len(set(ids)):
        parser.error("Manifest has duplicate building IDs")
    for building in buildings:
        if not building.pre_image.is_file() or not building.post_image.is_file():
            parser.error(f"Missing crop for {building.building_id}")

    result_path = args.output / "predictions.jsonl"
    if not args.pilot_only and result_path.exists():
        parser.error(f"{result_path} already exists; choose a new output directory")

    scorer = QwenChoiceScorer(args.model, args.revision, args.allow_cpu)
    if args.pilot_only:
        first = buildings[0]
        prediction = prediction_from_scores(scorer.pilot(first.pre_image, first.post_image))
        print(json.dumps({
            "model_revision": scorer.revision,
            "device": scorer.device,
            "answer_token_ids": scorer.token_ids,
            "prediction": prediction.label,
            "relative_confidence": prediction.confidence,
            "choice_probabilities": prediction.choice_probabilities,
        }, indent=2))
        return

    args.output.mkdir(parents=True, exist_ok=True)
    metadata_path = args.output / "run.json"
    metadata = {
        "model_id": scorer.model_id,
        "model_revision": scorer.revision,
        "device": scorer.device,
        "transformers_version": scorer.transformers_version,
        "torch_version": scorer.torch_version,
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "condition_set": "image-only, matching, nearest conflict, farthest conflict",
        "scoring": "next-token A/B/C/D logits normalized over four choices",
        "complete": False,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    with result_path.open("w", encoding="utf-8") as output:
        for index, building in enumerate(buildings, start=1):
            conditions = demo_conditions(building.true_label)
            baseline = prediction_from_scores(scorer.score_choices(
                building.pre_image, building.post_image, conditions[0].prompt
            ))
            for condition in conditions:
                prediction = baseline if condition.name == "image_only" else prediction_from_scores(
                    scorer.score_choices(building.pre_image, building.post_image, condition.prompt)
                )
                output.write(json.dumps(result_row(building, condition, baseline, prediction).to_record()) + "\n")
                output.flush()
            print(f"Scored {index}/{len(buildings)}: {building.building_id}", flush=True)

    metadata["answer_token_ids"] = scorer.token_ids
    metadata["complete"] = True
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Saved results to {result_path}")


if __name__ == "__main__":
    main()

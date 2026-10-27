"""Exercise xBD preparation and reporting with temporary test-only images."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from disaster_model.conditions import DAMAGE_LABELS
from disaster_model.data import read_manifest
from disaster_model.demo import demo_conditions
from disaster_model.results import result_row
from disaster_model.scoring import prediction_from_scores
from disaster_model.showcase import build_showcase
from disaster_model.xbd import prepare_sample


SUBTYPES = ("no-damage", "minor-damage", "major-damage", "destroyed")


class DemoPipelineTest(unittest.TestCase):
    def test_prepare_score_and_report(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            images = root / "xbd" / "train" / "images"
            labels = root / "xbd" / "train" / "labels"
            images.mkdir(parents=True)
            labels.mkdir(parents=True)
            for index, subtype in enumerate(SUBTYPES):
                tile = f"event_{index:08d}"
                pre_image = Image.new("RGB", (512, 512), "lightblue")
                post_image = Image.new("RGB", (512, 512), "tan")
                pre_image.putpixel((200, 200), (255, 0, 0))
                post_image.putpixel((250, 250), (0, 0, 255))
                pre_image.save(images / f"{tile}_pre_disaster.png")
                post_image.save(images / f"{tile}_post_disaster.png")
                annotation = {
                    "metadata": {"disaster_type": "test-event"},
                    "features": {"xy": [{
                        "wkt": "POLYGON ((230 230, 270 230, 270 270, 230 270, 230 230))",
                        "properties": {"uid": f"building-{index}", "subtype": subtype},
                    }]},
                }
                (labels / f"{tile}_post_disaster.json").write_text(json.dumps(annotation))
                pre_annotation = {
                    "features": {"xy": [{
                        "wkt": "POLYGON ((180 180, 220 180, 220 220, 180 220, 180 180))",
                        "properties": {"uid": f"building-{index}"},
                    }]},
                }
                (labels / f"{tile}_pre_disaster.json").write_text(json.dumps(pre_annotation))

            manifest = prepare_sample(root / "xbd", root / "processed", per_class=1)
            buildings = list(read_manifest(manifest))
            self.assertEqual([building.true_label for building in buildings], list(DAMAGE_LABELS))
            for building in buildings:
                with Image.open(building.pre_image) as pre, Image.open(building.post_image) as post:
                    self.assertEqual(pre.size, (256, 256))
                    self.assertEqual(post.size, pre.size)
                    self.assertEqual(pre.getpixel((128, 128)), (255, 0, 0))
                    self.assertEqual(post.getpixel((128, 128)), (0, 0, 255))

            results = root / "results"
            results.mkdir()
            predictions = results / "predictions.jsonl"
            with predictions.open("w") as output:
                for building in buildings:
                    baseline = prediction_from_scores({
                        choice: 4 if label == building.true_label else 0
                        for choice, label in zip("ABCD", DAMAGE_LABELS)
                    })
                    for condition in demo_conditions(building.true_label):
                        prediction = baseline
                        if condition.name == "conflicting_text":
                            prediction = prediction_from_scores({
                                choice: 4 if label == condition.text_claim else 0
                                for choice, label in zip("ABCD", DAMAGE_LABELS)
                            })
                        output.write(json.dumps(result_row(building, condition, baseline, prediction).to_record()) + "\n")
            (results / "run.json").write_text(json.dumps({
                "complete": False,
                "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                "model_id": "test-only-scorer",
                "model_revision": "0" * 40,
            }))
            page = results / "showcase.html"
            with self.assertRaisesRegex(ValueError, "incomplete"):
                build_showcase(manifest, predictions, page)
            run = json.loads((results / "run.json").read_text())
            run["complete"] = True
            (results / "run.json").write_text(json.dumps(run))
            build_showcase(manifest, predictions, page)
            content = page.read_text()
            self.assertIn("4/4", content)
            self.assertIn("8/8", content)
            self.assertIn("data:image/png;base64,", content)
            self.assertIn("Each row's A/B/C/D shares sum to 1", content)
            self.assertIn("A: no damage", content)

            records = [json.loads(line) for line in predictions.read_text().splitlines()]
            records[0]["choice_probabilities"]["A"] = 0.0
            predictions.write_text("".join(json.dumps(record) + "\n" for record in records))
            with self.assertRaisesRegex(ValueError, "Invalid A/B/C/D probabilities"):
                build_showcase(manifest, predictions, page)


if __name__ == "__main__":
    unittest.main()

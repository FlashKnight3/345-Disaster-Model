# Multimodal Disaster Damage Assessment

This project studies whether a vision-language model follows a text claim when it conflicts with satellite imagery. Each example is one building from xBD/xView2, represented by a pre-disaster crop, a post-disaster crop, and one of four damage labels. The planned model is a frozen, snapshot-pinned Qwen3-VL-8B-Instruct.

The repository is an initial scaffold. It defines the data record, paired experiment conditions, choice-score normalization, and result record; dataset preparation and Qwen inference will be added as their details are settled.

## Layout

```text
data/
  raw/                 Downloaded xBD files (not committed)
  processed/           Building crops and manifest (not committed)
docs/experiment-plan.md
results/               Predictions, metrics, and figures (not committed)
src/disaster_model/
  data.py              Building manifest format and reader
  conditions.py        Image-only and text-claim conditions
  model.py             Interface for scoring four answer choices
  scoring.py           Normalized choice scores and confidence
  results.py           Per-condition result record
```

## Data contract

`data/processed/manifest.jsonl` contains one JSON object per building. Paths to crops are relative to the manifest's directory. For example:

```json
{"building_id":"example-001","pre_image":"crops/example-001-pre.png","post_image":"crops/example-001-post.png","true_label":"major damage","disaster_type":"hurricane","split":"test"}
```

`building_id` should uniquely identify a building. `pre_image`, `post_image`, `true_label`, and `split` are required. `disaster_type` is optional but useful for failure analysis. The example above describes the format; no dataset files are included.

## Planned workflow

1. Put the xBD data in `data/raw/`, then create paired building crops and `data/processed/manifest.jsonl` from its polygons.
2. Pin the Qwen3-VL-8B-Instruct snapshot and run the same buildings through the image-only and text-claim conditions. Keep the images and A/B/C/D classification question fixed.
3. Score the four answer choices from model logits, normalize their scores with softmax, and save one result per building and condition in `results/`. The image-only result supplies the baseline prediction and confidence for every text condition.
4. Compute macro-F1, per-class precision/recall, confusion matrices, and prediction flip rates. See [the experiment plan](docs/experiment-plan.md).

Confidence means the normalized score of the chosen class **among the four answer choices**. It is not a calibrated probability of being correct. Before the full experiment, a small pilot must verify that A/B/C/D are scored consistently at the answer position. See [the experiment plan](docs/experiment-plan.md).

The model and dataset are intentionally not downloaded or run by this scaffold.

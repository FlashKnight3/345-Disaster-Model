# Midterm showcase runbook

The demo is a small paired experiment, not a full evaluation. It uses 12 buildings (three per xBD damage class) and four conditions per building: image-only, matching text, nearest conflicting claim, and farthest conflicting claim. That is 48 scored prompts. The showcase displays all 12 buildings, so examples cannot silently be selected after seeing model outputs.

## Needed

- Labeled xBD/xView2 image and JSON files from the [official dataset page](https://xview2.org/dataset) (registration may be required), with `images/` and `labels/` as sibling directories. The preparation script searches below the supplied root for `*_post_disaster.json` files.
- The project Mac (Apple M3 Pro, 36 GB unified memory) can try `Qwen/Qwen3-VL-8B-Instruct` through PyTorch's Metal (MPS) backend. Verify MPS in a normal Terminal with the command below before downloading model weights. The two-image pilot determines whether this setup has enough free memory and supports the model's operations. A hosted GPU is a fallback only if that pilot fails.
- Python 3.10 or newer and a recent Transformers version. See the [official Qwen3-VL documentation](https://huggingface.co/docs/transformers/model_doc/qwen3_vl) and [Apple Silicon guidance](https://huggingface.co/docs/transformers/apple_silicon).

## Commands

The 12-building sample is already prepared in `data/processed/manifest.jsonl`. Run these commands from the repository root on the Mac:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[demo]'
python -c 'import torch; print(torch.backends.mps.is_available())'
python scripts/run_demo.py --pilot-only
python scripts/run_demo.py
python scripts/build_showcase.py
open results/demo/showcase.html
```

The first pilot downloads the model weights, so it needs an internet connection and several gigabytes of free space. The prepared `data/processed/` sample is sufficient to rerun inference. To select different buildings later, extract the needed xBD files from the original training archive, run `python scripts/prepare_demo.py --xbd-root data/raw/train`, and then run `python scripts/preview_crops.py` before inference. A different manifest requires a new inference run.

The MPS check should print `True`. If it prints `False`, resolve that before model inference. The Mac has no separate NVIDIA GPU; its built-in Metal GPU uses the same 36 GB memory pool as the CPU.

Inspect `data/processed/contact_sheet.jpg` before inference. A yellow box marks the same target building in each image; it conveys location only, not the damage label. If the selection is unclear, rerun preparation with a different `--seed`, then inspect again. Do not present model outputs as results until this check passes.

The pilot checks the exact A/B/C/D token IDs in the formatted two-image prompt and scores the first building. It fails if any choice is not one stable answer token; that failure means the scoring implementation needs full-sequence scoring before the full run.

The full run writes `results/demo/predictions.jsonl` and `results/demo/run.json`. It refuses to overwrite an existing predictions file; choose another `--output` directory for a new run. The final `results/demo/showcase.html` embeds the crops and works offline.

## Presentation

1. State the question and show the fixed A/B/C/D choices.
2. Show one before/after pair and its image-only prediction and score.
3. Show the same pair with a matching and a conflicting claim.
4. In the current local run, image-only predictions matched 6 of 12 xBD labels. Qwen followed all 24 conflicting claims. All 22 claims whose label differed from the image-only prediction moved the prediction to the claim; the other two already matched the image-only prediction and cannot count as flips.
5. Point out a confident mistake: one no-damage building was classified as major damage with an image-only share of about 0.999. The four A/B/C/D shares sum to 1 for each row, but they are not calibrated probabilities of correctness.
6. Explain that this small sample demonstrates the pipeline and text sensitivity. It does not establish that lower image-only confidence causes more flips; nearly all baseline shares were high and the eligible conflicts all flipped.

The generated data and results directories are ignored by Git. Keep the model revision in `run.json` with any screenshots or exported results used in the showcase.

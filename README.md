# Disaster damage and conflicting text

This is our COMP 345 project on building damage assessment. We give Qwen3-VL a satellite image from before a disaster and one from after it, then ask for one of four damage labels: no damage, minor damage, major damage, or destroyed. The images and building labels come from xBD.

We want to see whether adding a written claim changes the model's answer, especially when the claim disagrees with the images.

## Midterm demo

For the midterm we selected 12 buildings, three from each damage class. A yellow box marks the same building in both images. Each building is scored four times: with images alone, with a claim matching the xBD label, and with two conflicting claims. The images and classification question stay the same across those runs.

We score A, B, C, and D using the model's logits, then apply softmax. The four numbers add up to 1 **within each prompt**. They show which answer the model prefers, not its actual probability of being correct.

In the current local run saved in `results/demo/`, the image-only answer matched the xBD label for 6 of 12 buildings. Qwen chose the claimed class in all 24 conflicting-claim prompts. In 22 of those prompts, the claim differed from the image-only prediction, so the answer flipped toward the claim; in the other two, the image-only answer already matched the claim. This is a small demo sample, so we are treating those numbers as observations rather than a general result about the model.

## Run or view the demo

The [IPython demo notebook](notebooks/disaster_demo.ipynb) walks through one
before/after image pair, the four prompts, and the results across all 12
buildings. It can use a completed local run or launch the model from a notebook
cell. Install the notebook tools with `python -m pip install -e '.[notebook]'`,
then open it with `jupyter lab notebooks/disaster_demo.ipynb`. For a new model
run, also install the `demo` extra and follow the notebook's data setup steps.

The notebook is tracked in Git; the xBD images, model weights, and generated
results are not. A fresh clone needs the xBD data and must run inference before
the result cells can show predictions.

On the Mac where we ran the experiment, open `results/demo/showcase.html` to see the images, predictions, and scores. The page works offline. From Terminal, in this folder:

```bash
open results/demo/showcase.html
```

The image crops and result files are ignored by Git, so a fresh clone will not include them. To prepare data and run the model yourself, follow [the midterm runbook](docs/midterm-demo.md). You will need the labeled xBD training archive from [xView2](https://xview2.org/dataset). The first model run also downloads Qwen3-VL weights.

The main commands are in `scripts/`: `prepare_demo.py` makes building crops, `run_demo.py` scores the four conditions, and `build_showcase.py` creates the HTML page. The code they use is in `src/disaster_model/`. Our broader study plan is in [docs/experiment-plan.md](docs/experiment-plan.md).

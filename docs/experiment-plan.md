# Experiment plan

Source: *Multimodal Disaster Damage Assessment* proposal by Rayaan Damani and Prad Chebolu.

## Question

When the visual evidence and a textual damage claim conflict, which does the frozen vision-language model follow?

The broader motivation is that satellite-derived and text-based evidence can capture different aspects of an event. [Sticher et al. (2026)](https://www.nature.com/articles/s41586-026-11004-6) discuss their complementary strengths in conflict research. Our experiment is narrower: its synthetic damage claims refer to the same building-level label as the imagery, and conflicting claims are deliberately false relative to the xBD label. The paper motivates careful handling of different information sources; it does not evaluate Qwen or this experiment.

## Unit of analysis

One building, identified by an xBD polygon and represented by matching pre-disaster and post-disaster crops. Use the ordered labels `no damage`, `minor damage`, `major damage`, and `destroyed`.

## Paired conditions

Run every selected building with the same two images and the same classification question:

1. Image-only baseline, without a damage claim.
2. A claim matching the true label.
3. Claims at each possible nonzero severity distance from the true label.

The code generates every possible claim for each building. This covers conflict distances of one, two, and three where possible while keeping the image pair fixed. The statement template is `The building shown has {label}.`

## Image-only confidence

Use a fixed answer format for every condition: `A` = no damage, `B` = minor damage, `C` = major damage, `D` = destroyed. Obtain comparable model log scores for the four choices at the answer position, then normalize those four scores with softmax. The predicted class is the highest-scoring choice, and its normalized score is the confidence measure. Do not ask the model to report a confidence percentage.

Qwen3-VL exposes language-model logits through its forward pass ([Hugging Face documentation](https://huggingface.co/docs/transformers/main/model_doc/qwen3_vl)). If each answer is one token in the exact prompt context, compare those four token logits. If an answer requires multiple tokens, score the full answer sequence instead. Run a small pilot to check tokenization, answer-position alignment, and consistent scoring before collecting the full results.

These normalized values express relative preference among the four choices. They are not guaranteed to be calibrated probabilities of correctness. Use the confidence from the **image-only** condition when testing whether low-confidence examples flip toward conflicting text.

Each result row records the true label, text claim, conflict distance, current prediction, four choice scores and normalized values, image-only prediction and confidence, and whether the prediction changed or moved toward the claim.

## Outputs to analyze

- Macro-F1 as the primary classification metric.
- Precision, recall, and confusion matrices by damage class.
- Rate of prediction changes from the image-only baseline, especially flips toward the text claim.
- Flip rates by conflict distance, image-only confidence, damage severity, and disaster type.
- Cases where conflicting text changes a correct image-only prediction to an incorrect one, or corrects an incorrect baseline prediction.

## Decisions for the full study

- Which xBD split and sampling scheme to use, and how to handle class imbalance.
- Crop padding and how to match each polygon to pre- and post-disaster imagery.
- Exact Qwen3-VL-8B-Instruct snapshot, inference settings, and output parsing.
- The exact answer-token or full-sequence scoring method after the pilot.
- Whether to run all possible claims per building or a balanced subset if compute is limited.

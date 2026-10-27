"""Qwen3-VL four-choice scorer, loaded only when inference is requested."""

from pathlib import Path
import re

from .conditions import ANSWER_CHOICES, CLASSIFICATION_QUESTION


def _messages(pre_image: Path, post_image: Path, prompt: str) -> list[dict]:
    return [{
        "role": "user",
        "content": [
            {"type": "text", "text": "Pre-disaster image:"},
            {"type": "image", "path": str(pre_image.resolve())},
            {"type": "text", "text": "Post-disaster image:"},
            {"type": "image", "path": str(post_image.resolve())},
            {"type": "text", "text": prompt},
        ],
    }]


def answer_token_ids(tokenizer, formatted_prompt: str) -> dict[str, int]:
    """Fail if A/B/C/D cannot each extend this prompt by exactly one token."""
    prefix = tokenizer.encode(formatted_prompt, add_special_tokens=False)
    ids = {}
    for choice in ANSWER_CHOICES:
        extended = tokenizer.encode(formatted_prompt + choice, add_special_tokens=False)
        if len(extended) != len(prefix) + 1 or extended[:-1] != prefix:
            raise ValueError(
                f"Answer {choice!r} is not one stable token in this prompt; "
                "use full-sequence scoring before running the demo"
            )
        ids[choice] = extended[-1]
    if len(set(ids.values())) != 4:
        raise ValueError("Answer choices do not have distinct token IDs")
    return ids


class QwenChoiceScorer:
    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-VL-8B-Instruct",
        revision: str | None = None,
        allow_cpu: bool = False,
    ) -> None:
        try:
            import torch
            import transformers
            from huggingface_hub import HfApi
            from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        except ImportError as exc:
            raise RuntimeError(
                "Install the demo dependencies on the inference machine: "
                "python -m pip install -e '.[demo]'"
            ) from exc

        self.device = (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
        if self.device == "cpu" and not allow_cpu:
            raise RuntimeError("No accelerator is available. Check MPS on a Mac, or pass --allow-cpu for a slow pilot.")
        self.model_id = model_id
        self.revision = (
            revision if revision and re.fullmatch(r"[0-9a-f]{40}", revision)
            else HfApi().model_info(model_id, revision=revision).sha
        )
        self.transformers_version = transformers.__version__
        self.torch_version = torch.__version__
        self.processor = AutoProcessor.from_pretrained(
            model_id, revision=self.revision,
            min_pixels=224 * 224, max_pixels=512 * 512,
        )
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id, revision=self.revision, dtype="auto",
            device_map="mps" if self.device == "mps" else "auto",
            attn_implementation="sdpa",
        ).eval()
        self.torch = torch
        self.token_ids: dict[str, int] | None = None

    def score_choices(self, pre_image: Path, post_image: Path, prompt: str) -> dict[str, float]:
        messages = _messages(pre_image, post_image, prompt)
        formatted = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        token_ids = answer_token_ids(self.processor.tokenizer, formatted)
        if self.token_ids is None:
            self.token_ids = token_ids
        elif token_ids != self.token_ids:
            raise ValueError("Answer token IDs changed between conditions")
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        )
        text_tail = self.processor.tokenizer.encode(formatted, add_special_tokens=False)[-5:]
        if inputs.input_ids[0, -len(text_tail):].tolist() != text_tail:
            raise ValueError("Processor and tokenizer disagree at the answer position")
        inputs = inputs.to(self.model.device)
        with self.torch.inference_mode():
            logits = self.model(**inputs, use_cache=False, logits_to_keep=1).logits[0, -1]
        return {choice: float(logits[token_id].item()) for choice, token_id in token_ids.items()}

    def pilot(self, pre_image: Path, post_image: Path) -> dict[str, float]:
        """Run one real two-image pass and verify the A-D answer position."""
        return self.score_choices(pre_image, post_image, CLASSIFICATION_QUESTION)

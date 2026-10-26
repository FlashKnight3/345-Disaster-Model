"""Interface to implement when the model snapshot and runtime are chosen."""

from pathlib import Path
from typing import Mapping, Protocol


class ChoiceScorer(Protocol):
    def score_choices(
        self, pre_image: Path, post_image: Path, prompt: str
    ) -> Mapping[str, float]:
        """Return comparable log scores for A, B, C, and D at the answer position.

        A single-token implementation can use the four answer-token logits.
        If an answer needs multiple tokens, score its entire token sequence.
        """
        ...

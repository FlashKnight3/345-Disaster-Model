"""Normalize four answer-choice scores into a relative confidence measure."""

from dataclasses import dataclass
from math import exp, isfinite
from typing import Mapping

from .conditions import ANSWER_CHOICES


@dataclass(frozen=True)
class Prediction:
    choice: str
    label: str
    confidence: float
    choice_log_scores: dict[str, float]
    choice_probabilities: dict[str, float]


def prediction_from_scores(log_scores: Mapping[str, float]) -> Prediction:
    """Softmax comparable A-D scores; confidence is the winning choice's share.

    The result is relative to these four choices and is not a calibrated
    probability that the classification is correct.
    """
    if set(log_scores) != set(ANSWER_CHOICES):
        raise ValueError("Scores must contain exactly the choices A, B, C, and D")
    scores = {choice: float(log_scores[choice]) for choice in ANSWER_CHOICES}
    if not all(isfinite(score) for score in scores.values()):
        raise ValueError("Choice scores must be finite")

    highest = max(scores.values())
    weights = {choice: exp(score - highest) for choice, score in scores.items()}
    total = sum(weights.values())
    probabilities = {choice: weight / total for choice, weight in weights.items()}
    choice = max(probabilities, key=probabilities.__getitem__)
    return Prediction(
        choice=choice,
        label=ANSWER_CHOICES[choice],
        confidence=probabilities[choice],
        choice_log_scores=scores,
        choice_probabilities=probabilities,
    )

"""One result record per building and experiment condition."""

from dataclasses import asdict, dataclass

from .conditions import Condition
from .data import Building
from .scoring import Prediction


@dataclass(frozen=True)
class ResultRow:
    building_id: str
    split: str
    disaster_type: str | None
    true_label: str
    condition: str
    text_claim: str | None
    conflict_distance: int | None
    predicted_choice: str
    predicted_label: str
    confidence: float
    choice_log_scores: dict[str, float]
    choice_probabilities: dict[str, float]
    image_only_prediction: str
    image_only_confidence: float
    prediction_changed: bool | None
    changed_toward_text: bool | None

    def to_record(self) -> dict:
        """Return a JSON-serializable record for a results JSONL file."""
        return asdict(self)


def result_row(
    building: Building,
    condition: Condition,
    baseline: Prediction,
    prediction: Prediction,
) -> ResultRow:
    """Join a condition's prediction to its image-only baseline."""
    has_text = condition.text_claim is not None
    return ResultRow(
        building_id=building.building_id,
        split=building.split,
        disaster_type=building.disaster_type,
        true_label=building.true_label,
        condition=condition.name,
        text_claim=condition.text_claim,
        conflict_distance=condition.conflict_distance,
        predicted_choice=prediction.choice,
        predicted_label=prediction.label,
        confidence=prediction.confidence,
        choice_log_scores=prediction.choice_log_scores,
        choice_probabilities=prediction.choice_probabilities,
        image_only_prediction=baseline.label,
        image_only_confidence=baseline.confidence,
        prediction_changed=(prediction.label != baseline.label) if has_text else None,
        changed_toward_text=(
            prediction.label == condition.text_claim
            and baseline.label != condition.text_claim
        ) if has_text else None,
    )

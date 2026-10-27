"""Fields saved for each prompt in a run."""

from .conditions import Condition
from .data import Building
from .scoring import Prediction


def result_row(
    building: Building,
    condition: Condition,
    baseline: Prediction,
    prediction: Prediction,
) -> dict:
    has_text = condition.text_claim is not None
    return {
        "building_id": building.building_id,
        "split": building.split,
        "disaster_type": building.disaster_type,
        "true_label": building.true_label,
        "condition": condition.name,
        "text_claim": condition.text_claim,
        "conflict_distance": condition.conflict_distance,
        "predicted_choice": prediction.choice,
        "predicted_label": prediction.label,
        "confidence": prediction.confidence,
        "choice_log_scores": prediction.choice_log_scores,
        "choice_probabilities": prediction.choice_probabilities,
        "image_only_prediction": baseline.label,
        "image_only_confidence": baseline.confidence,
        "prediction_changed": prediction.label != baseline.label if has_text else None,
        "changed_toward_text": (
            prediction.label == condition.text_claim
            and baseline.label != condition.text_claim
        ) if has_text else None,
    }

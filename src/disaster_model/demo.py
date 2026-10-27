"""Small paired condition set for the midterm showcase."""

from .conditions import Condition, conditions_for


def demo_conditions(true_label: str) -> tuple[Condition, ...]:
    """Baseline, matching claim, nearest conflict, and farthest conflict."""
    conditions = conditions_for(true_label)
    baseline = conditions[0]
    matching = next(item for item in conditions[1:] if item.conflict_distance == 0)
    conflicts = [item for item in conditions[1:] if item.conflict_distance]
    nearest = min(conflicts, key=lambda item: item.conflict_distance)
    farthest = max(conflicts, key=lambda item: item.conflict_distance)
    return baseline, matching, nearest, farthest

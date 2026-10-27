"""Fixed prompts and paired image/text conditions from the proposal."""

from dataclasses import dataclass


DAMAGE_LABELS = ("no damage", "minor damage", "major damage", "destroyed")
ANSWER_CHOICES = dict(zip("ABCD", DAMAGE_LABELS))
CLASSIFICATION_QUESTION = (
    "Compare the pre-disaster and post-disaster images of the building outlined in yellow. "
    "Classify its damage using these choices: A: no damage; B: minor damage; "
    "C: major damage; D: destroyed. Reply with only A, B, C, or D."
)


@dataclass(frozen=True)
class Condition:
    name: str
    text_claim: str | None
    conflict_distance: int | None
    prompt: str


def conditions_for(true_label: str) -> tuple[Condition, ...]:
    """Return the baseline and every possible damage claim for one building."""
    if true_label not in DAMAGE_LABELS:
        raise ValueError(f"Unknown damage label: {true_label!r}")

    conditions = [Condition("image_only", None, None, CLASSIFICATION_QUESTION)]
    for claim in DAMAGE_LABELS:
        distance = abs(DAMAGE_LABELS.index(true_label) - DAMAGE_LABELS.index(claim))
        name = "matching_text" if distance == 0 else "conflicting_text"
        statement = f"The building shown has {claim}."
        conditions.append(
            Condition(name, claim, distance, f"{statement}\n\n{CLASSIFICATION_QUESTION}")
        )
    return tuple(conditions)

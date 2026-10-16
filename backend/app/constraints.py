"""Deterministic fit, budget, and style checks. No LLM calls.

Style overlap is Jaccard similarity. Product tags and ``style_preference`` are
lowercased and split on non-alphanumeric runs (``mid-century`` and
``mid century`` share tokens). The score is ``|intersection| / |union|``,
or 0 when the union is empty. It is a score, not a hard reject.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from pydantic import BaseModel, Field

from app.retrieval import ProductCandidate
from app.schemas import Room

# Catalog category literals. Singular "sofa" / "chair" are not valid.
FIT_CATEGORIES = frozenset({"sofas", "chairs"})

_TOKEN = re.compile(r"[a-z0-9]+")


def usable_wall_lengths_m(room: Room) -> dict[str, float]:
    """Usable length of each wall in meters.

    North and south run along ``length_m``. East and west run along ``width_m``.
    Every opening is subtracted from its own wall. Lengths are clamped at 0.
    """

    lengths = {
        "north": float(room.dimensions.length_m),
        "south": float(room.dimensions.length_m),
        "east": float(room.dimensions.width_m),
        "west": float(room.dimensions.width_m),
    }
    for opening in room.openings:
        lengths[opening.wall] -= float(opening.width_m)
    return {wall: max(0.0, length) for wall, length in lengths.items()}


def tokenize_style(text: str) -> set[str]:
    """Lowercase alphanumeric tokens from a preference string or one tag."""

    return set(_TOKEN.findall(text.lower()))


def style_overlap(tags: Sequence[str], style_preference: str) -> float:
    """Jaccard overlap of product style tags and tokenized ``style_preference``."""

    preference = tokenize_style(style_preference)
    product: set[str] = set()
    for tag in tags:
        product |= tokenize_style(tag)
    union = preference | product
    if not union:
        return 0.0
    return len(preference & product) / len(union)


class ConstrainedCandidate(BaseModel):
    """A candidate that passed or failed, with the reason and its style score."""

    candidate: ProductCandidate
    reason: str = Field(min_length=1)
    style_overlap: float = Field(ge=0, le=1)


class ConstraintResult(BaseModel):
    """Accepted and rejected candidates.

    ``style_overlap_score`` is the mean Jaccard of ``accepted`` only. It is 0
    when nothing was accepted. Rejected rows still carry their own score.
    """

    accepted: list[ConstrainedCandidate]
    rejected: list[ConstrainedCandidate]
    style_overlap_score: float = Field(ge=0, le=1)


def _selected_ids(selected_so_far: Sequence[ProductCandidate | str]) -> set[str]:
    ids: set[str] = set()
    for item in selected_so_far:
        if isinstance(item, str):
            ids.add(item)
        else:
            ids.add(item.id)
    return ids


def _reject(candidate: ProductCandidate, reason: str, overlap: float) -> ConstrainedCandidate:
    return ConstrainedCandidate(candidate=candidate, reason=reason, style_overlap=overlap)


def apply_constraints(
    candidates: Sequence[ProductCandidate],
    room: Room,
    selected_so_far: Sequence[ProductCandidate | str],
    remaining_budget: float,
) -> ConstraintResult:
    """Accept candidates that are new, in budget, and (for sofas/chairs) fit.

    Checks run in order: already selected, then budget (``price_usd <=
    remaining_budget``), then fit. Fit applies only to ``sofas`` and ``chairs``:
    ``width_cm / 100 <= max(usable_wall_lengths_m)``. The first failure is the
    recorded reason. Input order is preserved inside each list.
    """

    walls = usable_wall_lengths_m(room)
    max_wall = max(walls.values()) if walls else 0.0
    chosen = _selected_ids(selected_so_far)
    accepted: list[ConstrainedCandidate] = []
    rejected: list[ConstrainedCandidate] = []

    for candidate in candidates:
        overlap = style_overlap(candidate.style_tags, room.style_preference)
        width_m = candidate.width_cm / 100.0
        if candidate.id in chosen:
            rejected.append(_reject(candidate, "already selected", overlap))
            continue
        if candidate.price_usd > remaining_budget:
            rejected.append(
                _reject(
                    candidate,
                    (
                        f"budget: price {candidate.price_usd:.2f} exceeds "
                        f"remaining {remaining_budget:.2f}"
                    ),
                    overlap,
                )
            )
            continue
        if candidate.category in FIT_CATEGORIES and width_m > max_wall:
            rejected.append(
                _reject(
                    candidate,
                    f"fit: width {width_m:.2f}m exceeds max usable wall {max_wall:.2f}m",
                    overlap,
                )
            )
            continue
        if candidate.category in FIT_CATEGORIES:
            reason = f"accepted: width {width_m:.2f}m fits max usable wall {max_wall:.2f}m"
        else:
            reason = f"accepted: {candidate.category} skips wall-fit check"
        accepted.append(_reject(candidate, reason, overlap))

    if accepted:
        score = sum(item.style_overlap for item in accepted) / len(accepted)
    else:
        score = 0.0
    return ConstraintResult(
        accepted=accepted,
        rejected=rejected,
        style_overlap_score=score,
    )

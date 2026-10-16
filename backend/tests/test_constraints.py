"""Deterministic fit, budget, and style-overlap constraints."""

import json
import sys
from pathlib import Path

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_ROOT.parent
sys.path.insert(0, str(BACKEND_ROOT))

from app.constraints import (  # noqa: E402
    FIT_CATEGORIES,
    apply_constraints,
    style_overlap,
    usable_wall_lengths_m,
)
from app.retrieval import ProductCandidate  # noqa: E402
from app.schemas import Room  # noqa: E402

SAMPLE_ROOM_PATH = REPO_ROOT / "data" / "rooms" / "sample_living_room.json"


def _room(
    *,
    length_m: float = 4.0,
    width_m: float = 3.0,
    openings: list[dict] | None = None,
    style_preference: str = "modern scandinavian",
    budget_usd: float = 2500,
) -> Room:
    return Room.model_validate(
        {
            "room_type": "living_room",
            "dimensions": {
                "length_m": length_m,
                "width_m": width_m,
                "height_m": 2.5,
            },
            "openings": openings or [],
            "existing_furniture": [],
            "style_preference": style_preference,
            "budget_usd": budget_usd,
            "user_notes": "",
        }
    )


def _candidate(**overrides) -> ProductCandidate:
    data = {
        "id": "sofa-1",
        "name": "Test Sofa",
        "category": "sofas",
        "style_tags": ["modern", "scandinavian"],
        "price_usd": 100.0,
        "width_cm": 200.0,
        "depth_cm": 90.0,
        "height_cm": 80.0,
        "material": "linen",
        "distance": 0.2,
        "score": 0.8,
        "document": "Test Sofa | sofas | modern, scandinavian | linen | A sofa.",
    }
    data.update(overrides)
    return ProductCandidate.model_validate(data)


def test_usable_wall_lengths_subtract_openings_on_that_wall_only():
    room = _room(
        length_m=4.0,
        width_m=3.0,
        openings=[
            {"type": "window", "wall": "north", "width_m": 1.0},
            {"type": "door", "wall": "north", "width_m": 0.5},
            {"type": "window", "wall": "east", "width_m": 0.5},
        ],
    )

    lengths = usable_wall_lengths_m(room)

    assert set(lengths) == {"north", "south", "east", "west"}
    assert lengths["north"] == pytest.approx(2.5)
    assert lengths["south"] == pytest.approx(4.0)
    assert lengths["east"] == pytest.approx(2.5)
    assert lengths["west"] == pytest.approx(3.0)


def test_sample_room_north_wall_loses_the_window():
    room = Room.model_validate(json.loads(SAMPLE_ROOM_PATH.read_text(encoding="utf-8")))

    lengths = usable_wall_lengths_m(room)

    assert lengths["north"] == pytest.approx(3.4)
    assert lengths["south"] == pytest.approx(5.2)
    assert lengths["east"] == pytest.approx(4.1)
    assert lengths["west"] == pytest.approx(4.1)


def test_openings_wider_than_the_wall_clamp_to_zero():
    room = _room(
        length_m=2.0,
        width_m=2.0,
        openings=[{"type": "garage", "wall": "north", "width_m": 2.5}],
    )

    lengths = usable_wall_lengths_m(room)

    assert lengths["north"] == 0.0
    assert lengths["south"] == 2.0
    assert lengths["east"] == 2.0
    assert lengths["west"] == 2.0


def test_fit_categories_are_catalog_sofa_and_chair_values():
    assert FIT_CATEGORIES == frozenset({"sofas", "chairs"})


def test_sofa_and_chair_fit_accepts_width_equal_to_max_usable_wall():
    room = _room(length_m=4.0, width_m=3.0)
    sofa = _candidate(id="sofa-fit", category="sofas", width_cm=400)
    chair = _candidate(id="chair-fit", category="chairs", width_cm=400, name="Chair")

    result = apply_constraints([sofa, chair], room, [], remaining_budget=1000)

    assert [item.candidate.id for item in result.accepted] == ["sofa-fit", "chair-fit"]
    assert result.rejected == []
    assert result.accepted[0].reason == "accepted: width 4.00m fits max usable wall 4.00m"
    assert result.accepted[1].reason == "accepted: width 4.00m fits max usable wall 4.00m"


def test_sofa_and_chair_fit_rejects_width_past_max_usable_wall():
    room = _room(length_m=4.0, width_m=3.0)
    sofa = _candidate(id="sofa-wide", category="sofas", width_cm=401)
    chair = _candidate(id="chair-wide", category="chairs", width_cm=401)

    result = apply_constraints([sofa, chair], room, [], remaining_budget=1000)

    assert result.accepted == []
    assert [item.candidate.id for item in result.rejected] == ["sofa-wide", "chair-wide"]
    assert result.rejected[0].reason == "fit: width 4.01m exceeds max usable wall 4.00m"
    assert result.rejected[1].reason == "fit: width 4.01m exceeds max usable wall 4.00m"


def test_non_seating_categories_skip_the_wall_fit_check():
    room = _room(length_m=2.0, width_m=2.0)
    rug = _candidate(
        id="rug-wide",
        name="Sofa-sized Rug",
        category="rugs",
        width_cm=900,
        style_tags=["modern"],
    )
    lamp = _candidate(id="lamp-1", category="lighting", width_cm=40, style_tags=["modern"])

    result = apply_constraints([rug, lamp], room, [], remaining_budget=500)

    assert [item.candidate.id for item in result.accepted] == ["rug-wide", "lamp-1"]
    assert result.rejected == []
    assert result.accepted[0].reason == "accepted: rugs skips wall-fit check"
    assert result.accepted[1].reason == "accepted: lighting skips wall-fit check"


def test_budget_rejects_price_above_remaining_and_accepts_equal():
    room = _room()
    fits = _candidate(id="priced-ok", price_usd=250)
    over = _candidate(id="priced-over", price_usd=250.01)

    result = apply_constraints([fits, over], room, [], remaining_budget=250)

    assert [item.candidate.id for item in result.accepted] == ["priced-ok"]
    assert [item.candidate.id for item in result.rejected] == ["priced-over"]
    assert result.rejected[0].reason == "budget: price 250.01 exceeds remaining 250.00"


def test_budget_is_reported_before_fit_when_both_fail():
    room = _room(length_m=2.0, width_m=2.0)
    both = _candidate(id="both", price_usd=50, width_cm=900)

    result = apply_constraints([both], room, [], remaining_budget=10)

    assert result.accepted == []
    assert result.rejected[0].reason == "budget: price 50.00 exceeds remaining 10.00"


def test_already_selected_ids_are_rejected_before_budget():
    room = _room()
    chosen = _candidate(id="sofa-1", price_usd=9999, width_cm=900)
    other = _candidate(id="sofa-2", name="Other", price_usd=1)

    by_model = apply_constraints([chosen, other], room, [chosen], remaining_budget=1)
    by_id = apply_constraints([chosen], room, ["sofa-1"], remaining_budget=1)

    assert [item.candidate.id for item in by_model.rejected] == ["sofa-1"]
    assert [item.candidate.id for item in by_model.accepted] == ["sofa-2"]
    assert by_model.rejected[0].reason == "already selected"
    assert by_id.rejected[0].reason == "already selected"


def test_style_overlap_is_jaccard_over_tokenized_tags_and_preference():
    # Extra tag makes Jaccard 2/3. Overlap coefficient and preference-containment
    # would both be 1.0, so this locks the metric.
    assert style_overlap(
        ["modern", "scandinavian", "minimal"],
        "modern scandinavian",
    ) == pytest.approx(2 / 3)
    assert style_overlap(["modern"], "modern scandinavian") == pytest.approx(0.5)
    assert style_overlap(["modern", "scandinavian"], "modern modern scandinavian") == pytest.approx(1.0)
    assert style_overlap(["industrial"], "modern scandinavian") == pytest.approx(0.0)
    assert style_overlap(["mid century", "modern"], "mid-century modern") == pytest.approx(1.0)
    assert style_overlap(["modern"], " ") == pytest.approx(0.0)


def test_style_overlap_score_is_the_mean_of_accepted_candidates():
    room = _room(style_preference="modern scandinavian")
    full = _candidate(id="full", style_tags=["modern", "scandinavian"])
    half = _candidate(id="half", style_tags=["modern"])
    none = _candidate(id="none", style_tags=["industrial"], price_usd=999)

    result = apply_constraints([full, half, none], room, [], remaining_budget=100)

    assert [item.candidate.id for item in result.accepted] == ["full", "half"]
    assert result.accepted[0].style_overlap == pytest.approx(1.0)
    assert result.accepted[1].style_overlap == pytest.approx(0.5)
    assert result.rejected[0].style_overlap == pytest.approx(0.0)
    assert result.style_overlap_score == pytest.approx(0.75)


def test_style_overlap_score_is_zero_when_nothing_is_accepted():
    room = _room()
    over = _candidate(price_usd=20, style_tags=["modern", "scandinavian"])

    result = apply_constraints([over], room, [], remaining_budget=0)

    assert result.accepted == []
    assert result.style_overlap_score == 0.0
    assert result.rejected[0].style_overlap == pytest.approx(1.0)


def test_constraint_order_follows_candidate_order():
    room = _room(length_m=4.0, width_m=3.0)
    first = _candidate(id="a", width_cm=100)
    rejected = _candidate(id="b", width_cm=900)
    third = _candidate(id="c", category="rugs", width_cm=900, style_tags=["modern"])

    result = apply_constraints([first, rejected, third], room, [], remaining_budget=500)

    assert [item.candidate.id for item in result.accepted] == ["a", "c"]
    assert [item.candidate.id for item in result.rejected] == ["b"]

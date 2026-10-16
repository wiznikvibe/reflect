"""Schema tests for room scans, catalog products, and grounded design plans."""

import json
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

BACKEND_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_ROOT.parent
sys.path.insert(0, str(BACKEND_ROOT))

from app.schemas import (  # noqa: E402
    ChatMessage,
    DesignPlan,
    DesignPlanItem,
    Product,
    Room,
)

SAMPLE_ROOM_PATH = REPO_ROOT / "data" / "rooms" / "sample_living_room.json"
CATALOG_PATH = REPO_ROOT / "data" / "catalog" / "living_room.json"

ALLOWED_CATEGORIES = {
    "sofas",
    "chairs",
    "coffee_tables",
    "side_tables",
    "lighting",
    "rugs",
}

VALID_PRODUCT = {
    "id": "sofa-oslo-linen",
    "name": "Oslo Three-Seat Sofa",
    "category": "sofas",
    "style_tags": ["modern", "scandinavian"],
    "dimensions": {"width_cm": 210, "depth_cm": 88, "height_cm": 80},
    "price_usd": 899,
    "material": "linen",
    "description": "Low-profile three-seat sofa in oatmeal linen.",
}


def test_room_parses_sample_scan_json():
    raw = json.loads(SAMPLE_ROOM_PATH.read_text(encoding="utf-8"))
    room = Room.model_validate(raw)

    assert room.room_type == "living_room"
    assert room.dimensions.length_m == 5.2
    assert room.dimensions.width_m == 4.1
    assert room.dimensions.height_m == 2.7
    assert len(room.openings) == 1
    opening = room.openings[0]
    assert opening.type == "window"
    assert opening.wall == "north"
    assert opening.width_m == 1.8
    assert room.existing_furniture == []
    assert room.style_preference == "modern scandinavian"
    assert room.budget_usd == 2500
    assert room.user_notes == "Need a sofa that faces the window and a coffee table"


def test_product_requires_catalog_fields():
    product = Product.model_validate(VALID_PRODUCT)
    assert product.id == "sofa-oslo-linen"
    assert product.category == "sofas"
    assert product.style_tags == ["modern", "scandinavian"]
    assert product.dimensions.width_cm == 210
    assert product.dimensions.depth_cm == 88
    assert product.dimensions.height_cm == 80
    assert product.price_usd == 899
    assert product.material == "linen"

    for key in (
        "id",
        "name",
        "category",
        "style_tags",
        "dimensions",
        "price_usd",
        "material",
        "description",
    ):
        broken = dict(VALID_PRODUCT)
        del broken[key]
        with pytest.raises(ValidationError):
            Product.model_validate(broken)

    bad_category = dict(VALID_PRODUCT)
    bad_category["category"] = "beds"
    with pytest.raises(ValidationError):
        Product.model_validate(bad_category)

    bad_dimensions = dict(VALID_PRODUCT)
    bad_dimensions["dimensions"] = {"depth_cm": 88, "height_cm": 80}
    with pytest.raises(ValidationError):
        Product.model_validate(bad_dimensions)


def test_catalog_covers_living_room_categories():
    raw_items = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    assert 50 <= len(raw_items) <= 70

    products = [Product.model_validate(item) for item in raw_items]
    ids = [product.id for product in products]
    assert len(ids) == len(set(ids))

    categories = {product.category for product in products}
    assert categories == ALLOWED_CATEGORIES

    styles = {tag for product in products for tag in product.style_tags}
    for expected in ("modern", "scandinavian", "mid-century", "industrial"):
        assert expected in styles

    prices = [product.price_usd for product in products]
    assert min(prices) < 200
    assert max(prices) > 1500


def test_design_plan_item_grounding_fields():
    item = DesignPlanItem(
        product_id="sofa-oslo-linen",
        role="primary seating",
        placement_note="Against the south wall, facing the north window",
        why_it_fits="Width fits the long wall and the tags overlap modern scandinavian.",
    )
    plan = DesignPlan(
        summary="A calm modern scandinavian living room under budget.",
        style_direction="modern scandinavian",
        items=[item],
        total_usd=899,
        budget_remaining_usd=1601,
        constraint_notes=["Sofa width is within the longest usable wall."],
    )

    assert plan.items[0].product_id == "sofa-oslo-linen"
    assert plan.items[0].role == "primary seating"
    assert plan.items[0].placement_note.startswith("Against the south wall")
    assert "scandinavian" in plan.items[0].why_it_fits
    assert plan.alternatives == []

    dumped = plan.model_dump()
    assert dumped["items"][0]["product_id"] == "sofa-oslo-linen"
    assert set(dumped["items"][0]) == {
        "product_id",
        "role",
        "placement_note",
        "why_it_fits",
    }

    with pytest.raises(ValidationError):
        DesignPlanItem(
            role="primary seating",
            placement_note="Against the south wall",
            why_it_fits="Looks nice",
        )

    with pytest.raises(ValidationError):
        DesignPlanItem(
            product_id="",
            role="primary seating",
            placement_note="Against the south wall",
            why_it_fits="Looks nice",
        )


def test_chat_message_round_trip():
    message = ChatMessage(role="user", content="Add a floor lamp under 200 dollars")
    assert message.model_dump() == {
        "role": "user",
        "content": "Add a floor lamp under 200 dollars",
    }

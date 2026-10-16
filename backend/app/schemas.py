"""Pydantic contracts for room scans, catalog products, and design plans."""

from typing import Literal

from pydantic import BaseModel, Field

Category = Literal[
    "sofas",
    "chairs",
    "coffee_tables",
    "side_tables",
    "lighting",
    "rugs",
]
Wall = Literal["north", "south", "east", "west"]


class Dimensions(BaseModel):
    """Room bounding box in meters."""

    length_m: float = Field(gt=0)
    width_m: float = Field(gt=0)
    height_m: float = Field(gt=0)


class Opening(BaseModel):
    """A door, window, or other opening that reduces usable wall length."""

    type: str = Field(min_length=1, description="Opening kind, such as window or door.")
    wall: Wall
    width_m: float = Field(gt=0)


class ProductDimensions(BaseModel):
    """Product size in centimeters."""

    width_cm: float = Field(gt=0)
    depth_cm: float = Field(gt=0)
    height_cm: float = Field(gt=0)


class Room(BaseModel):
    """Mock Nestingale room scan plus shopper preferences."""

    room_type: Literal["living_room"]
    dimensions: Dimensions
    openings: list[Opening] = Field(default_factory=list)
    existing_furniture: list[dict[str, object]] = Field(default_factory=list)
    style_preference: str = Field(min_length=1)
    budget_usd: float = Field(ge=0)
    user_notes: str = ""


class Product(BaseModel):
    """One shoppable living-room SKU."""

    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    category: Category
    style_tags: list[str] = Field(min_length=1)
    dimensions: ProductDimensions
    price_usd: float = Field(gt=0)
    material: str = Field(min_length=1)
    description: str = Field(min_length=1)


class DesignPlanItem(BaseModel):
    """A recommended SKU. product_id must be a catalog id, never invented."""

    product_id: str = Field(min_length=1)
    role: str = Field(min_length=1)
    placement_note: str = Field(min_length=1)
    why_it_fits: str = Field(min_length=1)


class DesignPlan(BaseModel):
    """Constraint-checked plan returned to the client."""

    summary: str = Field(min_length=1)
    style_direction: str = Field(min_length=1)
    items: list[DesignPlanItem]
    total_usd: float = Field(ge=0)
    budget_remaining_usd: float = Field(ge=0)
    constraint_notes: list[str]
    alternatives: list[DesignPlanItem] = Field(default_factory=list)


class ChatMessage(BaseModel):
    """One turn in a design-planning session."""

    role: Literal["user", "assistant", "system"]
    content: str

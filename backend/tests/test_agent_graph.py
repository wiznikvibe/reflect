"""LangGraph plan flow with a scripted planner. No live Gemini calls."""

import json
import sys
from pathlib import Path

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_ROOT.parent
sys.path.insert(0, str(BACKEND_ROOT))

from app.agent.graph import (  # noqa: E402
    DEFAULT_GEMINI_MODEL,
    MAX_PLAN_CANDIDATES,
    build_graph,
    build_plan_prompt,
)
from app.constraints import apply_constraints  # noqa: E402
from app.embeddings import FakeEmbedder  # noqa: E402
from app.ingest import ingest_catalog  # noqa: E402
from app.metrics import TokenUsage, estimate_cost_usd  # noqa: E402
from app.retrieval import ProductCandidate  # noqa: E402
from app.schemas import ChatMessage, DesignPlan, DesignPlanItem, Room  # noqa: E402

FORBIDDEN = "Do not invent SKUs or product IDs."


class ScriptedPlanner:
    """Returns a plan from the ids the graph claims are allowed."""

    def __init__(self, choose) -> None:
        self.choose = choose
        self.prompts: list[str] = []
        self.candidate_ids: list[list[str]] = []

    def __call__(self, prompt: str, candidate_ids: list[str]):
        self.prompts.append(prompt)
        self.candidate_ids.append(list(candidate_ids))
        items = [
            DesignPlanItem(
                product_id=product_id,
                role="piece",
                placement_note="Along the longest open wall.",
                why_it_fits="Listed in the allowed set and within the room brief.",
            )
            for product_id in self.choose(candidate_ids)
        ]
        plan = DesignPlan(
            summary="A grounded living-room plan using only allowed products.",
            style_direction="modern scandinavian",
            items=items,
            total_usd=1,
            budget_remaining_usd=0,
            constraint_notes=["model note"],
        )
        return plan, TokenUsage(prompt_tokens=120, completion_tokens=30)


def _room(budget_usd: float = 2500) -> Room:
    return Room.model_validate(
        {
            "room_type": "living_room",
            "dimensions": {"length_m": 5.2, "width_m": 4.1, "height_m": 2.7},
            "openings": [{"type": "window", "wall": "north", "width_m": 1.8}],
            "existing_furniture": [],
            "style_preference": "modern scandinavian",
            "budget_usd": budget_usd,
            "user_notes": "Need a sofa that faces the window",
        }
    )


def _catalog_item(product_id: str, *, category: str, price: float, width_cm: float = 100) -> dict:
    return {
        "id": product_id,
        "name": product_id.replace("-", " "),
        "category": category,
        "style_tags": ["modern", "scandinavian"],
        "dimensions": {"width_cm": width_cm, "depth_cm": 80, "height_cm": 70},
        "price_usd": price,
        "material": "linen",
        "description": f"{product_id} for the living room.",
    }


def _ingest(tmp_path: Path, products: list[dict]) -> Path:
    catalog = tmp_path / "catalog.json"
    catalog.write_text(json.dumps(products), encoding="utf-8")
    chroma_dir = tmp_path / "chroma"
    ingest_catalog(catalog, chroma_dir, rebuild=True, embedder=FakeEmbedder())
    return chroma_dir


def _kept_sofa() -> dict:
    return {
        "id": "keep-sofa",
        "name": "Kept Sofa",
        "category": "sofas",
        "style_tags": ["modern", "scandinavian"],
        "price_usd": 400.0,
        "width_cm": 180.0,
        "depth_cm": 88.0,
        "height_cm": 80.0,
        "material": "linen",
    }


def _invoke(tmp_path: Path, products: list[dict], planner: ScriptedPlanner, **state):
    chroma_dir = _ingest(tmp_path, products)
    graph = build_graph(planner, embedder=FakeEmbedder(), chroma_dir=chroma_dir)
    payload = {
        "room": _room(state.pop("budget_usd", 2500)).model_dump(),
        "message": state.pop("message"),
        "history": state.pop("history", []),
        "selected_products": state.pop("selected_products", []),
    }
    return graph.invoke(payload)


def test_default_model_and_candidate_cap():
    assert DEFAULT_GEMINI_MODEL == "gemini-2.5-flash"
    assert MAX_PLAN_CANDIDATES == 8


def test_env_example_points_at_gemini_25_flash():
    text = (REPO_ROOT / ".env.example").read_text(encoding="utf-8")
    assert "GEMINI_MODEL=gemini-2.5-flash" in text
    assert "GEMINI_MODEL=gemini-2.0-flash" not in text
    assert "GEMINI_INPUT_USD_PER_1M_TOKENS=0.30" in text
    assert "GEMINI_OUTPUT_USD_PER_1M_TOKENS=2.50" in text


def test_estimate_cost_uses_gemini_25_flash_rates(monkeypatch):
    monkeypatch.delenv("GEMINI_INPUT_USD_PER_1M_TOKENS", raising=False)
    monkeypatch.delenv("GEMINI_OUTPUT_USD_PER_1M_TOKENS", raising=False)
    assert estimate_cost_usd(1_000_000, 1_000_000) == pytest.approx(2.80)

    monkeypatch.setenv("GEMINI_INPUT_USD_PER_1M_TOKENS", "0.30")
    monkeypatch.setenv("GEMINI_OUTPUT_USD_PER_1M_TOKENS", "2.50")
    assert estimate_cost_usd(120, 30) == pytest.approx(120 / 1_000_000 * 0.30 + 30 / 1_000_000 * 2.50)


def test_graph_nodes_are_parse_retrieve_constrain_plan_respond(tmp_path):
    graph = build_graph(
        ScriptedPlanner(lambda ids: ids[:1]),
        embedder=FakeEmbedder(),
        chroma_dir=tmp_path / "unused",
    )
    edges = {(edge.source, edge.target) for edge in graph.get_graph().edges}
    assert edges == {
        ("__start__", "parse"),
        ("parse", "retrieve"),
        ("retrieve", "constrain"),
        ("constrain", "plan"),
        ("plan", "respond"),
        ("respond", "__end__"),
    }


def test_plan_prompt_forbids_inventing_skus():
    candidate = ProductCandidate(
        id="rug-new",
        name="New Rug",
        category="rugs",
        style_tags=["modern", "scandinavian"],
        price_usd=500,
        width_cm=160,
        depth_cm=230,
        height_cm=1,
        material="wool",
        distance=0.1,
        score=0.9,
    )
    prompt = build_plan_prompt(
        room=_room(1000),
        message="add a rug",
        history=[ChatMessage(role="user", content="keep the sofa")],
        selected_products=[_kept_sofa()],
        candidates=[candidate],
        remaining_budget=600,
        rejected_notes=["pricey-chair: budget"],
    )
    assert FORBIDDEN in prompt
    assert "allowed product IDs" in prompt
    assert "keep-sofa" in prompt
    assert "rug-new" in prompt
    assert "keep the sofa" in prompt
    assert "pricey-chair" in prompt
    assert "600" in prompt


def test_accepted_candidates_are_sliced_to_eight_before_plan(monkeypatch, tmp_path):
    products = [
        _catalog_item(f"sku-{index:02d}", category="rugs", price=40) for index in range(10)
    ]
    original_ids: list[str] = []

    def spy(candidates, room, selected_so_far, remaining_budget):
        result = apply_constraints(candidates, room, selected_so_far, remaining_budget)
        original_ids.extend(item.candidate.id for item in result.accepted)
        return result

    monkeypatch.setattr("app.agent.graph.apply_constraints", spy)
    monkeypatch.setattr(
        "app.retrieval.get_embedder",
        lambda: (_ for _ in ()).throw(AssertionError("live embedder")),
    )
    planner = ScriptedPlanner(lambda ids: list(ids))
    result = _invoke(tmp_path, products, planner, message="show me rugs", budget_usd=5000)

    assert len(original_ids) == 10
    assert planner.candidate_ids == [original_ids[:MAX_PLAN_CANDIDATES]]
    assert [item["product_id"] for item in result["plan"]["items"]] == original_ids[:8]
    assert result["plan"]["total_usd"] == pytest.approx(320)
    assert FORBIDDEN in planner.prompts[0]


def test_retrieval_query_includes_recent_chat_history(monkeypatch, tmp_path):
    products = [_catalog_item("sofa-a", category="sofas", price=400, width_cm=180)]
    seen: dict = {}

    def spy(query, room, filters, top_k, **kwargs):
        seen["query"] = query
        from app.retrieval import retrieve as real_retrieve

        return real_retrieve(query, room, filters, top_k, **kwargs)

    monkeypatch.setattr("app.agent.graph.retrieve", spy)
    marker = "navy sectional facing the window"
    padded = "pad-" + ("y" * 800)
    planner = ScriptedPlanner(lambda ids: ids[:1])
    result = _invoke(
        tmp_path,
        products,
        planner,
        message="make it cheaper",
        history=[
            {"role": "user", "content": "DROP_ME_ANCIENT"},
            {"role": "assistant", "content": "old reply to drop"},
            {"role": "user", "content": marker},
            {"role": "assistant", "content": "Suggested the oslo sofa."},
            {"role": "user", "content": "keep the linen"},
            {"role": "assistant", "content": padded},
        ],
    )

    query = seen["query"]
    assert query == result["retrieval_query"]
    assert "make it cheaper" in query
    assert marker in query
    assert "Suggested the oslo sofa." in query
    assert "keep the linen" in query
    assert "Need a sofa that faces the window" in query
    assert "DROP_ME_ANCIENT" not in query
    assert "old reply to drop" not in query
    assert padded not in query
    assert "pad-" in query


def test_replacement_price_filter_uses_full_room_budget(tmp_path):
    products = [
        _catalog_item("sofa-upgrade", category="sofas", price=900, width_cm=180),
        _catalog_item("sofa-too-much", category="sofas", price=1500, width_cm=180),
        _catalog_item("sofa-wide", category="sofas", price=400, width_cm=600),
    ]
    planner = ScriptedPlanner(lambda ids: ["sofa-upgrade"])
    result = _invoke(
        tmp_path,
        products,
        planner,
        message="replace the sofa",
        budget_usd=1000,
        selected_products=[_kept_sofa()],
    )

    assert result["filters"]["max_price_usd"] == pytest.approx(1000)
    assert result["remaining_budget"] == pytest.approx(600)
    assert planner.candidate_ids[0] == ["sofa-upgrade"]
    assert [item["product_id"] for item in result["plan"]["items"]] == ["sofa-upgrade"]
    assert result["plan"]["total_usd"] == pytest.approx(900)
    assert result["plan"]["budget_remaining_usd"] == pytest.approx(100)


def test_price_filter_uses_full_room_budget_and_repair_trims(monkeypatch, tmp_path):
    products = [
        _catalog_item("rug-a", category="rugs", price=500),
        _catalog_item("rug-b", category="rugs", price=500),
        _catalog_item("pricey-chair", category="chairs", price=650, width_cm=70),
    ]
    seen: dict = {}

    def spy(candidates, room, selected_so_far, remaining_budget):
        seen["remaining_budget"] = remaining_budget
        seen["selected"] = [
            item if isinstance(item, str) else item.id for item in selected_so_far
        ]
        return apply_constraints(candidates, room, selected_so_far, remaining_budget)

    monkeypatch.setattr("app.agent.graph.apply_constraints", spy)
    planner = ScriptedPlanner(lambda ids: ["keep-sofa", *ids])
    result = _invoke(
        tmp_path,
        products,
        planner,
        message="add a rug",
        budget_usd=1000,
        selected_products=[_kept_sofa()],
        history=[{"role": "user", "content": "we already picked the sofa"}],
    )

    assert seen["remaining_budget"] == pytest.approx(1000)
    assert "keep-sofa" in seen["selected"]
    assert result["filters"]["max_price_usd"] == pytest.approx(1000)
    assert result["remaining_budget"] == pytest.approx(600)
    assert "pricey-chair" not in planner.candidate_ids[0]
    assert set(planner.candidate_ids[0]) == {"rug-a", "rug-b"}
    item_ids = [item["product_id"] for item in result["plan"]["items"]]
    assert item_ids[0] == "keep-sofa"
    assert item_ids[1] == planner.candidate_ids[0][0]
    assert len(item_ids) == 2
    assert result["plan"]["total_usd"] == pytest.approx(900)
    assert result["plan"]["budget_remaining_usd"] == pytest.approx(100)
    assert any(planner.candidate_ids[0][1] in note for note in result["plan"]["constraint_notes"])
    assert "model note" in result["plan"]["constraint_notes"]
    assert result["plan"]["summary"] in result["response_text"]
    assert [product["id"] for product in result["products"]] == item_ids
    assert result["products"][0]["price_usd"] == pytest.approx(400)
    assert result["metrics"]["prompt_tokens"] == 120
    assert result["metrics"]["completion_tokens"] == 30
    assert result["metrics"]["approx_cost_usd"] == pytest.approx(estimate_cost_usd(120, 30))
    assert result["metrics"]["latency_ms"] >= 0


def test_invented_sku_is_removed_and_totals_use_catalog_prices(tmp_path):
    products = [_catalog_item("rug-real", category="rugs", price=100)]
    planner = ScriptedPlanner(lambda ids: ["fake-sku", ids[0]])
    result = _invoke(tmp_path, products, planner, message="add a rug", budget_usd=500)

    assert [item["product_id"] for item in result["plan"]["items"]] == ["rug-real"]
    assert result["plan"]["total_usd"] == pytest.approx(100)
    assert result["plan"]["budget_remaining_usd"] == pytest.approx(400)
    assert result["plan"]["total_usd"] != 1
    assert any("fake-sku" in note for note in result["plan"]["constraint_notes"])
    assert result["products"][0]["price_usd"] == pytest.approx(100)


def test_duplicate_plan_ids_are_kept_once(tmp_path):
    products = [_catalog_item("rug-real", category="rugs", price=100)]
    planner = ScriptedPlanner(lambda ids: [ids[0], ids[0]])
    result = _invoke(tmp_path, products, planner, message="add a rug", budget_usd=500)

    assert [item["product_id"] for item in result["plan"]["items"]] == ["rug-real"]
    assert result["plan"]["total_usd"] == pytest.approx(100)

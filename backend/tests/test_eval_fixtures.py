"""Eval fixtures match Room and catalog schemas. Mock eval stays offline."""

import re
import sys
from pathlib import Path

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_ROOT.parent
sys.path.insert(0, str(BACKEND_ROOT))
sys.path.insert(0, str(BACKEND_ROOT / "scripts"))

from app.ingest import default_catalog_path, load_catalog  # noqa: E402
from app.metrics import estimate_cost_usd  # noqa: E402
from app.schemas import DesignPlan, DesignPlanItem, Product, Room  # noqa: E402
from run_eval import (  # noqa: E402
    SCRIPTED_COMPLETION_TOKENS,
    SCRIPTED_PROMPT_TOKENS,
    EvalCase,
    accepted_violations,
    format_report,
    load_cases,
    percentile,
    plan_violations,
    retrieval_inputs,
    run,
    validate_fixtures,
)

QUERIES = REPO_ROOT / "data" / "evals" / "queries.json"
README = REPO_ROOT / "README.md"
SECRET = re.compile(r"AIza[0-9A-Za-z_\-]{20,}|sk-[A-Za-z0-9]{20,}")


def _room(**overrides) -> Room:
    raw = {
        "room_type": "living_room",
        "dimensions": {"length_m": 1.6, "width_m": 1.5, "height_m": 2.4},
        "openings": [],
        "existing_furniture": [],
        "style_preference": "modern scandinavian",
        "budget_usd": 100,
        "user_notes": "",
    }
    raw.update(overrides)
    return Room.model_validate(raw)


def _product(product_id: str, *, category: str = "sofas", price: float = 50, width_cm: float = 300) -> Product:
    return Product.model_validate(
        {
            "id": product_id,
            "name": product_id,
            "category": category,
            "style_tags": ["modern"],
            "dimensions": {"width_cm": width_cm, "depth_cm": 80, "height_cm": 70},
            "price_usd": price,
            "material": "linen",
            "description": "A catalog product used by the violation check.",
        }
    )


def _plan(product_id: str, total: float) -> DesignPlan:
    return DesignPlan(
        summary="A plan used to check budget and fit.",
        style_direction="modern",
        items=[
            DesignPlanItem(
                product_id=product_id,
                role="sofa",
                placement_note="On the long wall.",
                why_it_fits="Included so the violation helper has an item to measure.",
            )
        ],
        total_usd=total,
        budget_remaining_usd=0,
        constraint_notes=[],
    )


def test_eval_fixtures_validate_against_schemas():
    cases = load_cases(QUERIES)
    validate_fixtures(cases, repo_root=REPO_ROOT, catalog_path=default_catalog_path())

    assert 5 <= len(cases) <= 10
    catalog_ids = {product.id for product in load_catalog(default_catalog_path())}
    for case in cases:
        assert isinstance(case, EvalCase)
        assert set(case.must_respect) <= {"budget", "fit"}
        assert set(case.relevant_product_ids) <= catalog_ids
        room = Room.model_validate_json((REPO_ROOT / case.room).read_text(encoding="utf-8"))
        assert room.room_type == "living_room"
        assert room.budget_usd >= 0


def test_readme_documents_gemini_25_flash_and_has_no_api_key():
    text = README.read_text(encoding="utf-8")
    assert "GEMINI_MODEL=gemini-2.5-flash" in text
    assert "gemini-2.0-flash" not in text
    assert "GOOGLE_API_KEY" in text
    assert ".env.example" in text
    assert "python -m app.ingest --rebuild" in text
    assert "uvicorn app.main:app" in text
    assert "npm run dev" in text
    assert "FakeEmbedder" in text
    assert SECRET.search(text) is None
    bundled = "\n".join(
        path.read_text(encoding="utf-8")
        for path in [
            QUERIES,
            BACKEND_ROOT / "scripts" / "run_eval.py",
            * (REPO_ROOT / "data" / "evals").rglob("*.json"),
        ]
    )
    assert SECRET.search(bundled) is None


def test_retrieval_inputs_follow_the_agent_parse_filters():
    room = Room.model_validate_json(
        (REPO_ROOT / "data" / "rooms" / "sample_living_room.json").read_text(encoding="utf-8")
    )

    sofa_query, sofa_filters = retrieval_inputs("a sofa", room)
    assert sofa_query == f"a sofa {room.user_notes}"
    assert sofa_filters.categories == ["sofas"]
    assert sofa_filters.max_price_usd == pytest.approx(room.budget_usd)
    assert sofa_filters.max_width_cm == pytest.approx(520)

    lamp_query, lamp_filters = retrieval_inputs("a brass floor lamp", room)
    assert lamp_query.startswith("a brass floor lamp ")
    assert lamp_filters.categories == ["lighting"]
    assert lamp_filters.max_width_cm is None
    assert lamp_filters.max_price_usd == pytest.approx(room.budget_usd)


def test_constraint_helpers_flag_budget_and_fit():
    room = _room(budget_usd=100)
    wide = _product("sofa-wide", width_cm=300, price=50)
    catalog = {wide.id: wide}

    over_budget = plan_violations(_plan("sofa-wide", 250), room, catalog, ["budget", "fit"])
    assert "budget" in over_budget
    assert "fit:sofa-wide" in over_budget

    within = plan_violations(_plan("sofa-wide", 80), room, catalog, ["budget"])
    assert within == []

    from app.retrieval import ProductCandidate

    candidate = ProductCandidate(
        id="sofa-wide",
        name="Wide Sofa",
        category="sofas",
        style_tags=["modern"],
        price_usd=150,
        width_cm=300,
        depth_cm=80,
        height_cm=70,
        material="linen",
        distance=0.1,
        score=0.9,
    )
    assert accepted_violations([candidate], room, ["budget", "fit"]) == [
        "budget:sofa-wide",
        "fit:sofa-wide",
    ]


def test_percentile_interpolates():
    assert percentile([10, 20, 30, 40], 50) == pytest.approx(25)
    assert percentile([10, 20, 30, 40], 95) == pytest.approx(38.5)
    assert percentile([7], 95) == pytest.approx(7)


def test_mock_eval_reports_metrics_without_a_live_model():
    report = run(mock=True, retrieval_only=False)
    text = format_report(report)

    assert report.mode == "mock"
    assert len(report.results) == len(load_cases(QUERIES))
    assert 0 <= report.hit_rate <= 1
    assert report.constraint_violation_rate == 0
    assert report.latency_p50_ms <= report.latency_p95_ms
    assert report.mean_approx_cost_usd == pytest.approx(
        estimate_cost_usd(SCRIPTED_PROMPT_TOKENS, SCRIPTED_COMPLETION_TOKENS)
    )
    assert "retrieval hit rate:" in text
    assert "constraint-violation rate:" in text
    assert "latency p50 ms:" in text
    assert "latency p95 ms:" in text
    assert "mean approx cost usd:" in text


def test_mock_retrieval_only_has_zero_cost_and_no_violations():
    report = run(mock=True, retrieval_only=True)

    assert report.mode == "mock-retrieval-only"
    assert report.constraint_violation_rate == 0
    assert report.mean_approx_cost_usd == 0
    assert all(result.approx_cost_usd == 0 for result in report.results)

"""Offline eval for the design-planning agent.

Reports retrieval hit rate, constraint-violation rate, p50/p95 latency, and
mean approximate Gemini cost over ``data/evals/queries.json``.

``--mock`` uses ``FakeEmbedder`` and a scripted planner so CI never calls
Gemini or downloads EmbeddingGemma. ``--retrieval-only`` skips the planner
and scores the retrieved and constrained candidate set.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

BACKEND_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_ROOT.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.agent.graph import TOP_K, build_graph, detect_categories  # noqa: E402
from app.constraints import FIT_CATEGORIES, apply_constraints, usable_wall_lengths_m  # noqa: E402
from app.embeddings import Embedder, FakeEmbedder, get_embedder  # noqa: E402
from app.ingest import default_catalog_path, default_chroma_dir, get_collection, ingest_catalog, load_catalog  # noqa: E402
from app.metrics import TokenUsage, UsageMetrics  # noqa: E402
from app.retrieval import ProductCandidate, RetrievalFilters, retrieve  # noqa: E402
from app.schemas import DesignPlan, DesignPlanItem, Product, Room  # noqa: E402

SCRIPTED_PROMPT_TOKENS = 120
SCRIPTED_COMPLETION_TOKENS = 30
MustRespect = Literal["budget", "fit"]


class EvalCase(BaseModel):
    """One canned query. ``room`` is a path relative to the repo root."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, pattern=r"^[a-z0-9][a-z0-9-]*$")
    room: str = Field(min_length=1)
    query: str = Field(min_length=1)
    relevant_product_ids: list[str] = Field(min_length=1)
    must_respect: list[MustRespect] = Field(min_length=1)

    @field_validator("room")
    @classmethod
    def room_is_repo_relative(cls, value: str) -> str:
        path = Path(value)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("room must be a repo-relative path")
        return value

    @field_validator("relevant_product_ids", "must_respect")
    @classmethod
    def unique_entries(cls, value: list[str]) -> list[str]:
        if any(not item for item in value):
            raise ValueError("entries must be non-empty")
        if len(value) != len(set(value)):
            raise ValueError("duplicate entries")
        return value


@dataclass(frozen=True)
class CaseResult:
    """Metrics for a single fixture."""

    case_id: str
    hit: bool
    violations: tuple[str, ...]
    latency_ms: float
    approx_cost_usd: float


@dataclass(frozen=True)
class EvalReport:
    """Aggregate metrics plus one row per fixture."""

    mode: str
    results: tuple[CaseResult, ...]

    @property
    def hit_rate(self) -> float:
        return _mean(1.0 if result.hit else 0.0 for result in self.results)

    @property
    def constraint_violation_rate(self) -> float:
        return _mean(1.0 if result.violations else 0.0 for result in self.results)

    @property
    def latency_p50_ms(self) -> float:
        return percentile([result.latency_ms for result in self.results], 50)

    @property
    def latency_p95_ms(self) -> float:
        return percentile([result.latency_ms for result in self.results], 95)

    @property
    def mean_approx_cost_usd(self) -> float:
        return _mean(result.approx_cost_usd for result in self.results)


class ScriptedPlanner:
    """Chooses the first allowed ids. Token counts are fixed so cost is stable."""

    def __call__(self, prompt: str, candidate_ids: list[str]) -> tuple[DesignPlan, TokenUsage]:
        del prompt
        items = [
            DesignPlanItem(
                product_id=product_id,
                role="piece",
                placement_note="From the allowed candidate list.",
                why_it_fits="Passed retrieval and the deterministic checks.",
            )
            for product_id in candidate_ids[:2]
        ]
        plan = DesignPlan(
            summary="Scripted plan for eval. No live model call.",
            style_direction="eval",
            items=items,
            total_usd=0,
            budget_remaining_usd=0,
            constraint_notes=[],
        )
        return plan, TokenUsage(
            prompt_tokens=SCRIPTED_PROMPT_TOKENS,
            completion_tokens=SCRIPTED_COMPLETION_TOKENS,
        )


def default_queries_path() -> Path:
    return REPO_ROOT / "data" / "evals" / "queries.json"


def load_cases(path: Path | None = None) -> list[EvalCase]:
    """Parse the fixture file. Refs are checked in ``validate_fixtures``."""

    query_path = default_queries_path() if path is None else Path(path)
    raw = json.loads(query_path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        items = raw.get("cases")
    else:
        items = raw
    if not isinstance(items, list):
        raise ValueError(f"eval file must contain a cases list: {query_path}")
    cases = [EvalCase.model_validate(item) for item in items]
    if not 5 <= len(cases) <= 10:
        raise ValueError(f"expected 5–10 eval cases, found {len(cases)}")
    ids = [case.id for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate eval case ids")
    return cases


def resolve_repo_path(ref: str, repo_root: Path | None = None) -> Path:
    """Resolve a repo-relative fixture path and reject escapes."""

    root = (REPO_ROOT if repo_root is None else repo_root).resolve()
    path = (root / ref).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"path escapes the repo: {ref}")
    return path


def validate_fixtures(
    cases: list[EvalCase],
    *,
    repo_root: Path | None = None,
    catalog_path: Path | None = None,
) -> None:
    """Check room files parse as ``Room`` and relevant ids exist in the catalog."""

    catalog = load_catalog(default_catalog_path() if catalog_path is None else catalog_path)
    known = {product.id for product in catalog}
    for case in cases:
        missing = [product_id for product_id in case.relevant_product_ids if product_id not in known]
        if missing:
            joined = ", ".join(missing)
            raise ValueError(f"{case.id}: unknown product ids: {joined}")
        room_path = resolve_repo_path(case.room, repo_root)
        if not room_path.is_file():
            raise ValueError(f"{case.id}: room fixture not found: {case.room}")
        Room.model_validate_json(room_path.read_text(encoding="utf-8"))


def retrieval_inputs(message: str, room: Room) -> tuple[str, RetrievalFilters]:
    """First-turn query and filters, matching the agent's parse node."""

    categories = detect_categories(message)
    filters: dict[str, Any] = {"max_price_usd": float(room.budget_usd)}
    if categories:
        filters["categories"] = categories
    if categories and set(categories) <= FIT_CATEGORIES:
        max_wall_m = max(usable_wall_lengths_m(room).values(), default=0.0)
        max_width_cm = max_wall_m * 100
        if max_width_cm > 0:
            filters["max_width_cm"] = max_width_cm
    parts = [message.strip()]
    notes = room.user_notes.strip()
    if notes:
        parts.append(notes)
    return " ".join(parts).strip(), RetrievalFilters.model_validate(filters)


def percentile(values: list[float], pct: float) -> float:
    """Linear percentile. ``pct`` is in 0–100."""

    if not values:
        raise ValueError("percentile of an empty list")
    if not 0 <= pct <= 100:
        raise ValueError("percentile must be between 0 and 100")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * (pct / 100.0)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return ordered[low]
    weight = rank - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def plan_violations(
    plan: DesignPlan,
    room: Room,
    catalog: dict[str, Product],
    must_respect: list[MustRespect],
) -> list[str]:
    """Budget and fit breaks on a grounded plan.

    ``budget`` compares ``total_usd`` with the room budget. ``fit`` rejects a
    sofa or chair wider than the longest usable wall. An unknown id cannot be
    shown to fit, so it counts as a fit violation when fit is required.
    """

    found: list[str] = []
    if "budget" in must_respect and plan.total_usd > room.budget_usd + 1e-6:
        found.append("budget")
    if "fit" in must_respect:
        max_wall = max(usable_wall_lengths_m(room).values(), default=0.0)
        for item in plan.items:
            product = catalog.get(item.product_id)
            if product is None or (
                product.category in FIT_CATEGORIES
                and product.dimensions.width_cm / 100 > max_wall + 1e-9
            ):
                found.append(f"fit:{item.product_id}")
    return found


def accepted_violations(
    candidates: list[ProductCandidate],
    room: Room,
    must_respect: list[MustRespect],
) -> list[str]:
    """Same budget and fit rules, applied to candidates the constraint step kept."""

    found: list[str] = []
    max_wall = max(usable_wall_lengths_m(room).values(), default=0.0)
    for candidate in candidates:
        if "budget" in must_respect and candidate.price_usd > room.budget_usd + 1e-6:
            found.append(f"budget:{candidate.id}")
        if (
            "fit" in must_respect
            and candidate.category in FIT_CATEGORIES
            and candidate.width_cm / 100 > max_wall + 1e-9
        ):
            found.append(f"fit:{candidate.id}")
    return found


def is_hit(retrieved_ids: list[str], relevant_ids: list[str]) -> bool:
    """True when top-k retrieval contains at least one labeled product."""

    return bool(set(retrieved_ids) & set(relevant_ids))


def format_report(report: EvalReport) -> str:
    """Plain-text CLI report."""

    lines = [
        f"mode: {report.mode}",
        f"cases: {len(report.results)}",
        f"retrieval hit rate: {report.hit_rate:.3f}",
        f"constraint-violation rate: {report.constraint_violation_rate:.3f}",
        f"latency p50 ms: {report.latency_p50_ms:.3f}",
        f"latency p95 ms: {report.latency_p95_ms:.3f}",
        f"mean approx cost usd: {report.mean_approx_cost_usd:.6f}",
        "",
    ]
    for result in report.results:
        violations = ",".join(result.violations) if result.violations else "none"
        lines.append(
            f"{result.case_id}  hit={'yes' if result.hit else 'no'}  "
            f"violations={violations}  latency_ms={result.latency_ms:.3f}  "
            f"cost_usd={result.approx_cost_usd:.6f}"
        )
    return "\n".join(lines)


def run(
    *,
    mock: bool = False,
    retrieval_only: bool = False,
    queries_path: Path | None = None,
    chroma_dir: Path | None = None,
) -> EvalReport:
    """Run every fixture. ``mock`` builds a temporary FakeEmbedder index."""

    cases = load_cases(queries_path)
    validate_fixtures(cases)
    catalog = {product.id: product for product in load_catalog(default_catalog_path())}
    mode = _mode_name(mock, retrieval_only)
    if mock:
        with tempfile.TemporaryDirectory() as tmp:
            index = Path(tmp) / "chroma"
            embedder = FakeEmbedder()
            ingest_catalog(default_catalog_path(), index, rebuild=True, embedder=embedder)
            results = _run_cases(
                cases,
                catalog,
                embedder=embedder,
                chroma_dir=index,
                retrieval_only=retrieval_only,
                planner=ScriptedPlanner(),
            )
    else:
        index = default_chroma_dir() if chroma_dir is None else Path(chroma_dir)
        _require_index(index)
        results = _run_cases(
            cases,
            catalog,
            embedder=get_embedder(),
            chroma_dir=index,
            retrieval_only=retrieval_only,
            planner=None,
        )
    return EvalReport(mode=mode, results=tuple(results))


def main(argv: list[str] | None = None) -> int:
    """CLI entry: ``python scripts/run_eval.py --mock`` from ``backend/``."""

    parser = argparse.ArgumentParser(description="Run the Nestingale design-planning eval set.")
    parser.add_argument(
        "--mock",
        action="store_true",
        help="FakeEmbedder and a scripted planner. No network and no API key.",
    )
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Score retrieval and constraints without calling a planner.",
    )
    parser.add_argument("--queries", type=Path, default=None)
    parser.add_argument(
        "--chroma-dir",
        type=Path,
        default=None,
        help="Persisted Chroma directory for a live run. Ignored with --mock.",
    )
    args = parser.parse_args(argv)
    try:
        report = run(
            mock=args.mock,
            retrieval_only=args.retrieval_only,
            queries_path=args.queries,
            chroma_dir=args.chroma_dir,
        )
    except (ValueError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(format_report(report))
    return 0


def _mode_name(mock: bool, retrieval_only: bool) -> str:
    if mock and retrieval_only:
        return "mock-retrieval-only"
    if mock:
        return "mock"
    if retrieval_only:
        return "retrieval-only"
    return "live"


def _mean(values) -> float:
    items = list(values)
    if not items:
        return 0.0
    return sum(items) / len(items)


def _require_index(chroma_dir: Path) -> None:
    if not chroma_dir.exists():
        raise RuntimeError(
            f"Chroma index not found at {chroma_dir}. "
            "From backend/, run: python -m app.ingest --rebuild"
        )
    try:
        count = get_collection(chroma_dir).count()
    except Exception as exc:
        raise RuntimeError(f"Could not open Chroma index at {chroma_dir}: {exc}") from exc
    if count <= 0:
        raise RuntimeError(
            f"Chroma index at {chroma_dir} is empty. "
            "From backend/, run: python -m app.ingest --rebuild"
        )


def _load_room(case: EvalCase) -> Room:
    return Room.model_validate_json(resolve_repo_path(case.room).read_text(encoding="utf-8"))


def _run_cases(
    cases: list[EvalCase],
    catalog: dict[str, Product],
    *,
    embedder: Embedder,
    chroma_dir: Path,
    retrieval_only: bool,
    planner: ScriptedPlanner | None,
) -> list[CaseResult]:
    graph = None
    if not retrieval_only:
        if planner is None:
            _load_env()
            from app.agent.graph import GeminiPlanner

            planner = GeminiPlanner()
        graph = build_graph(planner, embedder=embedder, chroma_dir=chroma_dir, top_k=TOP_K)

    results: list[CaseResult] = []
    for case in cases:
        room = _load_room(case)
        if retrieval_only:
            results.append(_retrieval_only_case(case, room, embedder, chroma_dir))
            continue
        assert graph is not None
        started = time.perf_counter()
        state = graph.invoke(
            {
                "room": room.model_dump(),
                "message": case.query,
                "history": [],
                "selected_products": [],
            }
        )
        elapsed_ms = (time.perf_counter() - started) * 1000
        retrieved = [str(item.get("id", "")) for item in state.get("candidates") or []]
        plan = DesignPlan.model_validate(state["plan"])
        metrics = UsageMetrics.model_validate(state["metrics"])
        results.append(
            CaseResult(
                case_id=case.id,
                hit=is_hit(retrieved, case.relevant_product_ids),
                violations=tuple(plan_violations(plan, room, catalog, case.must_respect)),
                latency_ms=metrics.latency_ms if metrics.latency_ms > 0 else round(elapsed_ms, 3),
                approx_cost_usd=metrics.approx_cost_usd,
            )
        )
    return results


def _retrieval_only_case(
    case: EvalCase,
    room: Room,
    embedder: Embedder,
    chroma_dir: Path,
) -> CaseResult:
    started = time.perf_counter()
    query, filters = retrieval_inputs(case.query, room)
    found = retrieve(query, room, filters, TOP_K, chroma_dir=chroma_dir, embedder=embedder)
    constrained = apply_constraints(found, room, [], float(room.budget_usd))
    latency_ms = round((time.perf_counter() - started) * 1000, 3)
    accepted = [row.candidate for row in constrained.accepted]
    return CaseResult(
        case_id=case.id,
        hit=is_hit([item.id for item in found], case.relevant_product_ids),
        violations=tuple(accepted_violations(accepted, room, case.must_respect)),
        latency_ms=latency_ms,
        approx_cost_usd=0.0,
    )


def _load_env() -> None:
    """Load the repo ``.env`` for a live planner run. The key is never printed."""

    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")
    import os

    if not os.environ.get("GOOGLE_API_KEY"):
        raise RuntimeError(
            "GOOGLE_API_KEY is missing. Put it in .env (leave .env.example blank) "
            "or re-run with --mock."
        )


if __name__ == "__main__":
    raise SystemExit(main())

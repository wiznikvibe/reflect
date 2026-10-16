"""Fixed design graph: parse → retrieve → constrain → plan → respond.

The planner may only use ids from the current selection and the constrained
candidates. After the model answers, totals are recomputed from catalog prices
and any combination that exceeds the room budget is trimmed in listed order.
"""

from __future__ import annotations

import os
import re
import time
from typing import Any, Protocol

from langgraph.graph import END, START, StateGraph

from app.agent.state import AgentState
from app.constraints import FIT_CATEGORIES, apply_constraints, style_overlap, usable_wall_lengths_m
from app.embeddings import Embedder
from app.metrics import TokenUsage, UsageMetrics, estimate_cost_usd
from app.retrieval import ProductCandidate, RetrievalFilters, retrieve
from app.schemas import ChatMessage, DesignPlan, DesignPlanItem, Room

DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
MAX_PLAN_CANDIDATES = 8
TOP_K = 12

_CATEGORY_PATTERNS: tuple[tuple[str, str], ...] = (
    ("coffee_tables", r"\bcoffee tables?\b"),
    ("side_tables", r"\b(?:side|end) tables?\b"),
    ("sofas", r"\b(?:sofas?|couches?|sectionals?)\b"),
    ("chairs", r"\bchairs?\b"),
    ("lighting", r"\b(?:lamps?|lighting|lights?)\b"),
    ("rugs", r"\brugs?\b"),
)


class Planner(Protocol):
    """Structured planner. ``candidate_ids`` are the ids already written into the prompt."""

    def __call__(self, prompt: str, candidate_ids: list[str]) -> tuple[DesignPlan, TokenUsage]:
        """Return a plan and token counts. The server may drop ids and rewrite totals."""


def detect_categories(message: str) -> list[str]:
    """Catalog categories named in the current user message, in catalog order."""

    found: list[str] = []
    for category, pattern in _CATEGORY_PATTERNS:
        if re.search(pattern, message, flags=re.IGNORECASE):
            found.append(category)
    return found


def _history_lines(history: list[ChatMessage | dict]) -> list[str]:
    lines: list[str] = []
    for message in history:
        if isinstance(message, ChatMessage):
            role, content = message.role, message.content
        else:
            role, content = str(message.get("role", "user")), str(message.get("content", ""))
        lines.append(f"{role}: {content}")
    return lines


def _candidate_line(candidate: ProductCandidate, room: Room) -> str:
    overlap = style_overlap(candidate.style_tags, room.style_preference)
    styles = ", ".join(candidate.style_tags)
    return (
        f"- {candidate.id} | {candidate.name} | {candidate.category} | "
        f"{candidate.price_usd:.2f} USD | "
        f"{candidate.width_cm:.0f}x{candidate.depth_cm:.0f}x{candidate.height_cm:.0f} cm | "
        f"material {candidate.material} | styles {styles} | style overlap {overlap:.2f}"
    )


def build_plan_prompt(
    *,
    room: Room,
    message: str,
    history: list[ChatMessage | dict],
    selected_products: list[dict],
    candidates: list[ProductCandidate],
    remaining_budget: float,
    rejected_notes: list[str],
) -> str:
    """Ground the planner. The forbidden-SKU sentence is part of the contract."""

    selected_lines = [
        (
            f"- {product['id']} | {product.get('name', product['id'])} | "
            f"{float(product['price_usd']):.2f} USD"
        )
        for product in selected_products
    ]
    candidate_lines = [_candidate_line(candidate, room) for candidate in candidates]
    history_lines = _history_lines(history)
    rejected = "\n".join(f"- {note}" for note in rejected_notes) or "- none"
    lines = [
        "You are a living-room design planner.",
        "Do not invent SKUs or product IDs. Use only these allowed product IDs.",
        "Do not invent prices. The server overwrites total_usd and budget_remaining_usd.",
        "Return the full selection. Repeat an already selected id to keep it.",
        "Items and alternatives must use allowed product IDs only.",
        "",
        f"Room: {room.model_dump_json()}",
        f"Room budget USD: {room.budget_usd:.2f}",
        f"Remaining budget for new items USD: {remaining_budget:.2f}",
        f"Current message: {message}",
        "History:",
        *(history_lines or ["- none"]),
        "",
        "Already selected:",
        *(selected_lines or ["- none"]),
        "",
        "Allowed new candidates:",
        *(candidate_lines or ["- none"]),
        "",
        "Rejected this turn:",
        rejected,
    ]
    return "\n".join(lines)


def _product_from_candidate(candidate: ProductCandidate) -> dict[str, Any]:
    return {
        "id": candidate.id,
        "name": candidate.name,
        "category": candidate.category,
        "style_tags": list(candidate.style_tags),
        "price_usd": float(candidate.price_usd),
        "width_cm": float(candidate.width_cm),
        "depth_cm": float(candidate.depth_cm),
        "height_cm": float(candidate.height_cm),
        "material": candidate.material,
    }


def _index_products(
    selected_products: list[dict],
    candidates: list[ProductCandidate],
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for product in selected_products:
        indexed[str(product["id"])] = dict(product)
    for candidate in candidates:
        indexed[candidate.id] = _product_from_candidate(candidate)
    return indexed


def _product_view(info: dict[str, Any], item: DesignPlanItem) -> dict[str, Any]:
    view = dict(info)
    view["id"] = item.product_id
    view["price_usd"] = float(info["price_usd"])
    view["role"] = item.role
    view["placement_note"] = item.placement_note
    view["why_it_fits"] = item.why_it_fits
    return view


def repair_plan(
    plan: DesignPlan,
    allowed: dict[str, dict[str, Any]],
    budget_usd: float,
) -> tuple[DesignPlan, list[dict[str, Any]]]:
    """Drop unknown ids and stop adding items once the room budget is exceeded.

    Items are kept in model order. The first items win. Totals always come from
    catalog prices, never from the model.
    """

    notes = list(plan.constraint_notes)
    kept: list[DesignPlanItem] = []
    products: list[dict[str, Any]] = []
    seen: set[str] = set()
    running = 0.0
    for item in plan.items:
        info = allowed.get(item.product_id)
        if info is None:
            notes.append(f"dropped invented SKU {item.product_id}")
            continue
        if item.product_id in seen:
            notes.append(f"dropped duplicate {item.product_id}")
            continue
        price = float(info["price_usd"])
        if running + price > budget_usd + 1e-6:
            notes.append(
                f"dropped {item.product_id}: price {price:.2f} would exceed budget {budget_usd:.2f}"
            )
            continue
        running += price
        seen.add(item.product_id)
        kept.append(item)
        products.append(_product_view(info, item))

    alternatives: list[DesignPlanItem] = []
    for item in plan.alternatives:
        if item.product_id not in allowed:
            notes.append(f"dropped invented SKU {item.product_id}")
            continue
        if item.product_id in seen:
            continue
        seen.add(item.product_id)
        alternatives.append(item)

    total = round(running, 2)
    remaining = round(max(0.0, budget_usd - total), 2)
    repaired = plan.model_copy(
        update={
            "items": kept,
            "alternatives": alternatives,
            "total_usd": total,
            "budget_remaining_usd": remaining,
            "constraint_notes": notes,
        }
    )
    return repaired, products


def render_response(plan: DesignPlan) -> str:
    """Assistant text streamed as token events."""

    lines = [plan.summary, ""]
    if plan.items:
        for item in plan.items:
            lines.append(f"- {item.role}: {item.product_id} — {item.why_it_fits}")
    else:
        lines.append("No products fit the budget and the allowed catalog.")
    if plan.constraint_notes:
        lines.extend(["", "Notes: " + "; ".join(plan.constraint_notes)])
    lines.append(
        f"Total ${plan.total_usd:.2f}. Remaining ${plan.budget_remaining_usd:.2f}."
    )
    return "\n".join(lines)


def _usage_from_raw(raw: Any) -> TokenUsage:
    meta = getattr(raw, "usage_metadata", None) or {}
    if not isinstance(meta, dict):
        try:
            meta = dict(meta)
        except (TypeError, ValueError):
            meta = {}
    prompt = meta.get("input_tokens", meta.get("prompt_token_count", 0)) or 0
    completion = meta.get("output_tokens", meta.get("candidates_token_count", 0)) or 0
    return TokenUsage(prompt_tokens=int(prompt), completion_tokens=int(completion))


class GeminiPlanner:
    """Gemini Flash structured output. The client is created on the first call."""

    def __init__(self, model: str | None = None, api_key: str | None = None) -> None:
        self.model = model
        self.api_key = api_key

    def __call__(self, prompt: str, candidate_ids: list[str]) -> tuple[DesignPlan, TokenUsage]:
        del candidate_ids
        from langchain_google_genai import ChatGoogleGenerativeAI

        model = self.model or os.environ.get("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
        api_key = self.api_key if self.api_key is not None else os.environ.get("GOOGLE_API_KEY")
        llm = ChatGoogleGenerativeAI(
            model=model,
            google_api_key=api_key,
            temperature=0,
            thinking_budget=0,
        )
        structured = llm.with_structured_output(
            DesignPlan,
            method="json_schema",
            include_raw=True,
        )
        result = structured.invoke(prompt)
        parsed = result["parsed"]
        if isinstance(parsed, DesignPlan):
            plan = parsed
        elif parsed is not None:
            plan = DesignPlan.model_validate(parsed)
        else:
            error = result.get("parsing_error")
            raise RuntimeError(f"Gemini returned no design plan: {error}")
        return plan, _usage_from_raw(result.get("raw"))


def build_graph(
    planner: Planner,
    *,
    embedder: Embedder,
    chroma_dir: str | Any,
    top_k: int = TOP_K,
):
    """Compile the fixed five-node graph. ``planner`` is called only from plan."""

    def parse_node(state: AgentState) -> dict:
        room = Room.model_validate(state["room"])
        message = str(state.get("message", "")).strip()
        selected = list(state.get("selected_products") or [])
        spent = sum(float(product["price_usd"]) for product in selected)
        remaining = max(0.0, float(room.budget_usd) - spent)
        categories = detect_categories(message)
        filters: dict[str, Any] = {"max_price_usd": remaining}
        if categories:
            filters["categories"] = categories
        if categories and set(categories) <= FIT_CATEGORIES:
            max_wall_m = max(usable_wall_lengths_m(room).values(), default=0.0)
            max_width_cm = max_wall_m * 100
            if max_width_cm > 0:
                filters["max_width_cm"] = max_width_cm
        notes = room.user_notes.strip()
        retrieval_query = f"{message} {notes}".strip() if notes else message
        return {
            "retrieval_query": retrieval_query,
            "filters": filters,
            "remaining_budget": remaining,
            "started_at": time.perf_counter(),
        }

    def retrieve_node(state: AgentState) -> dict:
        room = Room.model_validate(state["room"])
        filters = RetrievalFilters.model_validate(state.get("filters") or {})
        found = retrieve(
            state.get("retrieval_query", ""),
            room,
            filters,
            top_k,
            chroma_dir=chroma_dir,
            embedder=embedder,
        )
        return {"candidates": [item.model_dump() for item in found]}

    def constrain_node(state: AgentState) -> dict:
        room = Room.model_validate(state["room"])
        candidates = [ProductCandidate.model_validate(item) for item in state.get("candidates") or []]
        selected_ids = [str(product["id"]) for product in state.get("selected_products") or []]
        result = apply_constraints(
            candidates,
            room,
            selected_ids,
            float(state.get("remaining_budget", room.budget_usd)),
        )
        accepted = result.accepted[:MAX_PLAN_CANDIDATES]
        rejected_notes = [f"{item.candidate.id}: {item.reason}" for item in result.rejected]
        if len(result.accepted) > MAX_PLAN_CANDIDATES:
            rejected_notes.append(
                f"sliced accepted candidates from {len(result.accepted)} to {MAX_PLAN_CANDIDATES}"
            )
        return {
            "accepted": [item.model_dump() for item in accepted],
            "rejected_notes": rejected_notes,
        }

    def plan_node(state: AgentState) -> dict:
        room = Room.model_validate(state["room"])
        accepted_rows = state.get("accepted") or []
        candidates = [
            ProductCandidate.model_validate(row["candidate"]) for row in accepted_rows
        ]
        selected = list(state.get("selected_products") or [])
        prompt = build_plan_prompt(
            room=room,
            message=str(state.get("message", "")),
            history=list(state.get("history") or []),
            selected_products=selected,
            candidates=candidates,
            remaining_budget=float(state.get("remaining_budget", room.budget_usd)),
            rejected_notes=list(state.get("rejected_notes") or []),
        )
        drafted, usage = planner(prompt, [candidate.id for candidate in candidates])
        if not isinstance(drafted, DesignPlan):
            drafted = DesignPlan.model_validate(drafted)
        if not isinstance(usage, TokenUsage):
            usage = TokenUsage.model_validate(usage)
        allowed = _index_products(selected, candidates)
        repaired, products = repair_plan(drafted, allowed, float(room.budget_usd))
        slice_notes = [
            note
            for note in state.get("rejected_notes") or []
            if str(note).startswith("sliced ")
        ]
        if slice_notes:
            repaired = repaired.model_copy(
                update={"constraint_notes": [*repaired.constraint_notes, *slice_notes]}
            )
        return {
            "plan": repaired.model_dump(),
            "products": products,
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
        }

    def respond_node(state: AgentState) -> dict:
        plan = DesignPlan.model_validate(state["plan"])
        started = float(state.get("started_at", time.perf_counter()))
        latency_ms = round(max(0.0, (time.perf_counter() - started) * 1000), 3)
        prompt_tokens = int(state.get("prompt_tokens", 0))
        completion_tokens = int(state.get("completion_tokens", 0))
        metrics = UsageMetrics(
            latency_ms=latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            approx_cost_usd=estimate_cost_usd(prompt_tokens, completion_tokens),
        )
        return {
            "response_text": render_response(plan),
            "metrics": metrics.model_dump(),
        }

    graph = StateGraph(AgentState)
    graph.add_node("parse", parse_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("constrain", constrain_node)
    graph.add_node("plan", plan_node)
    graph.add_node("respond", respond_node)
    graph.add_edge(START, "parse")
    graph.add_edge("parse", "retrieve")
    graph.add_edge("retrieve", "constrain")
    graph.add_edge("constrain", "plan")
    graph.add_edge("plan", "respond")
    graph.add_edge("respond", END)
    return graph.compile()

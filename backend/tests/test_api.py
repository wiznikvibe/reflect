"""FastAPI session and SSE tests. Gemini and EmbeddingGemma stay mocked."""

import asyncio
import json
import sys
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

BACKEND_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_ROOT.parent
sys.path.insert(0, str(BACKEND_ROOT))

from app.agent.state import SessionStore  # noqa: E402
from app.embeddings import FakeEmbedder  # noqa: E402


class RecordingEmbedder(FakeEmbedder):
    def __init__(self) -> None:
        super().__init__()
        self.queries: list[str] = []

    def embed_query(self, text: str) -> list[float]:
        self.queries.append(text)
        return super().embed_query(text)
from app.ingest import ingest_catalog  # noqa: E402
from app.main import create_app  # noqa: E402
from app.metrics import TokenUsage, UsageMetrics, estimate_cost_usd  # noqa: E402
from app.schemas import ChatMessage, DesignPlan, DesignPlanItem, Room  # noqa: E402

SAMPLE_ROOM = json.loads((REPO_ROOT / "data" / "rooms" / "sample_living_room.json").read_text())
SUMMARY = "A calm modern scandinavian living room anchored by one sofa that faces the window."


class RecordingPlanner:
    def __init__(self) -> None:
        self.candidate_ids: list[list[str]] = []

    def __call__(self, prompt: str, candidate_ids: list[str]):
        self.candidate_ids.append(list(candidate_ids))
        product_id = candidate_ids[0]
        plan = DesignPlan(
            summary=SUMMARY,
            style_direction="modern scandinavian",
            items=[
                DesignPlanItem(
                    product_id=product_id,
                    role="sofa",
                    placement_note="Facing the north window.",
                    why_it_fits="Fits the longest wall and the modern scandinavian brief.",
                )
            ],
            total_usd=1,
            budget_remaining_usd=0,
            constraint_notes=[],
        )
        return plan, TokenUsage(prompt_tokens=100, completion_tokens=40)


def _catalog() -> list[dict]:
    return [
        {
            "id": "sofa-oslo",
            "name": "Oslo Three-Seat Sofa",
            "category": "sofas",
            "style_tags": ["modern", "scandinavian"],
            "dimensions": {"width_cm": 210, "depth_cm": 88, "height_cm": 80},
            "price_usd": 899,
            "material": "oatmeal linen",
            "description": "Low-profile three-seat sofa.",
        },
        {
            "id": "pricey-sofa",
            "name": "Pricey Sofa",
            "category": "sofas",
            "style_tags": ["modern"],
            "dimensions": {"width_cm": 200, "depth_cm": 90, "height_cm": 80},
            "price_usd": 4000,
            "material": "leather",
            "description": "A sofa over the sample budget.",
        },
        {
            "id": "rug-fjord",
            "name": "Fjord Wool Rug",
            "category": "rugs",
            "style_tags": ["scandinavian"],
            "dimensions": {"width_cm": 160, "depth_cm": 230, "height_cm": 1},
            "price_usd": 240,
            "material": "wool",
            "description": "Flatweave wool rug.",
        },
    ]


def _app(tmp_path: Path, monkeypatch, *, ingest: bool = True):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_INPUT_USD_PER_1M_TOKENS", "0.30")
    monkeypatch.setenv("GEMINI_OUTPUT_USD_PER_1M_TOKENS", "2.50")

    def explode(*args, **kwargs):
        raise AssertionError("Gemini client constructed")

    monkeypatch.setattr("langchain_google_genai.ChatGoogleGenerativeAI", explode)
    chroma_dir = tmp_path / "chroma"
    embedder = RecordingEmbedder()
    if ingest:
        catalog = tmp_path / "catalog.json"
        catalog.write_text(json.dumps(_catalog()), encoding="utf-8")
        ingest_catalog(catalog, chroma_dir, rebuild=True, embedder=embedder)
    planner = RecordingPlanner()
    store = SessionStore()
    app = create_app(
        store=store,
        embedder=embedder,
        chroma_dir=chroma_dir,
        planner=planner,
    )
    return app, store, planner, embedder


def _run(coro):
    return asyncio.run(coro)


def _parse_sse(raw: str) -> list[dict[str, str]]:
    events: list[dict[str, str]] = []
    event_name: str | None = None
    data_lines: list[str] = []

    def flush() -> None:
        nonlocal event_name, data_lines
        if event_name is None and not data_lines:
            return
        events.append({"event": event_name or "message", "data": "\n".join(data_lines)})
        event_name = None
        data_lines = []

    for line in raw.splitlines():
        if line == "":
            flush()
        elif line.startswith("event:"):
            event_name = line.split(":", 1)[1].strip()
        elif line.startswith("data:"):
            data_lines.append(line.split(":", 1)[1].strip())
    flush()
    return events


def test_session_store_keeps_plan_products_and_metrics():
    store = SessionStore()
    room = Room.model_validate(SAMPLE_ROOM)
    first = store.create(room)
    second = store.create(room)
    assert first != second
    store.append_message(first, ChatMessage(role="user", content="hello"))
    plan = DesignPlan(
        summary="Stored plan.",
        style_direction="modern scandinavian",
        items=[
            DesignPlanItem(
                product_id="sofa-oslo",
                role="sofa",
                placement_note="Facing the window.",
                why_it_fits="In budget.",
            )
        ],
        total_usd=899,
        budget_remaining_usd=1601,
        constraint_notes=[],
    )
    metrics = UsageMetrics(
        latency_ms=12,
        prompt_tokens=2,
        completion_tokens=3,
        approx_cost_usd=0.01,
    )
    store.set_plan(first, plan, products=[{"id": "sofa-oslo", "price_usd": 899}], metrics=metrics)
    saved = store.get(first)
    assert saved.room.budget_usd == 2500
    assert saved.messages[0].content == "hello"
    assert saved.plan is not None and saved.plan.summary == "Stored plan."
    assert saved.products[0]["price_usd"] == 899
    assert saved.metrics is not None and saved.metrics.prompt_tokens == 2
    with pytest.raises(KeyError):
        store.get("missing")


def test_health_and_session_create_without_live_models(tmp_path, monkeypatch):
    app, _, _, _ = _app(tmp_path, monkeypatch, ingest=False)

    async def scenario():
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            health = await client.get("/api/health", headers={"Origin": "http://localhost:5173"})
            created = await client.post("/api/sessions", json=SAMPLE_ROOM)
            missing = await client.get("/api/sessions/missing/plan")
            invalid = await client.post("/api/sessions", json={"room_type": "kitchen"})
        return health, created, missing, invalid

    health, created, missing, invalid = _run(scenario())
    assert health.status_code == 200
    body = health.json()
    assert body == {"status": "degraded", "chroma_ready": False, "embedder_ready": True}
    assert health.headers["access-control-allow-origin"] == "http://localhost:5173"
    assert not (tmp_path / "chroma").exists()
    assert created.status_code == 200
    assert created.json()["session_id"]
    assert missing.status_code == 404
    assert invalid.status_code == 422


def test_chat_sse_then_get_plan(tmp_path, monkeypatch):
    app, store, planner, embedder = _app(tmp_path, monkeypatch)

    async def scenario():
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            created = await client.post("/api/sessions", json=SAMPLE_ROOM)
            session_id = created.json()["session_id"]
            before = await client.get(f"/api/sessions/{session_id}/plan")
            async with client.stream(
                "POST",
                f"/api/sessions/{session_id}/chat",
                json={"message": "Need a sofa"},
            ) as response:
                assert response.status_code == 200
                assert response.headers["content-type"].startswith("text/event-stream")
                raw = (await response.aread()).decode()
            empty = await client.post(
                f"/api/sessions/{session_id}/chat",
                json={"message": ""},
            )
            unknown = await client.post(
                "/api/sessions/missing/chat",
                json={"message": "Need a sofa"},
            )
            follow = await client.get(f"/api/sessions/{session_id}/plan")
            health = await client.get("/api/health")
        return session_id, before, raw, empty, unknown, follow, health

    session_id, before, raw, empty, unknown, follow, health = _run(scenario())
    assert before.json() == {"plan": None, "products": [], "metrics": None}
    events = _parse_sse(raw)
    kinds = [event["event"] for event in events]
    plan_at = kinds.index("plan")
    assert kinds[0] == "token"
    assert kinds[-1] == "done"
    assert plan_at < len(kinds) - 1
    assert kinds[plan_at + 1] == "done"
    assert set(kinds[:plan_at]) == {"token"}
    joined = "".join(json.loads(event["data"])["text"] for event in events if event["event"] == "token")
    assert SUMMARY in joined
    payload = json.loads(events[plan_at]["data"])
    assert set(payload) == {"plan", "products", "metrics"}
    plan = DesignPlan.model_validate(payload["plan"])
    assert plan.items[0].product_id == "sofa-oslo"
    assert plan.total_usd == pytest.approx(899)
    assert plan.budget_remaining_usd == pytest.approx(1601)
    assert payload["products"][0]["id"] == "sofa-oslo"
    assert payload["products"][0]["price_usd"] == pytest.approx(899)
    assert payload["products"][0]["why_it_fits"] == plan.items[0].why_it_fits
    metrics = payload["metrics"]
    assert set(metrics) == {"latency_ms", "prompt_tokens", "completion_tokens", "approx_cost_usd"}
    assert metrics["prompt_tokens"] == 100
    assert metrics["completion_tokens"] == 40
    assert metrics["approx_cost_usd"] == pytest.approx(estimate_cost_usd(100, 40))
    assert metrics["latency_ms"] >= 0
    assert planner.candidate_ids == [["sofa-oslo"]]
    assert any("coffee table" in query for query in embedder.queries)
    assert follow.json() == payload
    saved = store.get(session_id)
    assert [message.role for message in saved.messages] == ["user", "assistant"]
    assert saved.messages[0].content == "Need a sofa"
    assert SUMMARY in saved.messages[1].content
    assert empty.status_code == 422
    assert unknown.status_code == 404
    assert health.json()["status"] == "ok"
    assert health.json()["chroma_ready"] is True
    assert health.json()["embedder_ready"] is True

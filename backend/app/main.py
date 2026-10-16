"""FastAPI sessions and SSE chat for the design-planning agent."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.agent.graph import GeminiPlanner, Planner, build_graph
from app.agent.state import SessionStore
from app.embeddings import Embedder, get_embedder
from app.ingest import default_chroma_dir, get_collection
from app.metrics import UsageMetrics
from app.schemas import ChatMessage, DesignPlan, Room

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

VITE_ORIGIN = "http://localhost:5173"
_TOKEN_CHARS = 48


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)


def _chroma_ready(chroma_dir: Path) -> bool:
    if not chroma_dir.exists():
        return False
    try:
        return get_collection(chroma_dir).count() > 0
    except Exception:
        return False


def _sse(event: str, payload: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, separators=(',', ':'))}\n\n"


def iter_token_chunks(text: str, size: int = _TOKEN_CHARS) -> list[str]:
    if text == "":
        return [""]
    return [text[index : index + size] for index in range(0, len(text), size)]


def create_app(
    *,
    store: SessionStore | None = None,
    embedder: Embedder | None = None,
    chroma_dir: str | Path | None = None,
    planner: Planner | None = None,
) -> FastAPI:
    """Build the API. Inject ``planner`` and ``embedder`` in tests to avoid live models."""

    if store is None:
        store = SessionStore()
    if chroma_dir is None:
        chroma_dir = default_chroma_dir()
    chroma_path = Path(chroma_dir)
    if embedder is None:
        embedder = get_embedder()
    if planner is None:
        planner = GeminiPlanner()

    graph = build_graph(planner, embedder=embedder, chroma_dir=chroma_path)
    app = FastAPI(title="Nestingale design planning")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[VITE_ORIGIN],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.store = store
    app.state.graph = graph
    app.state.embedder = embedder
    app.state.chroma_dir = chroma_path

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        index_ready = _chroma_ready(chroma_path)
        embedder_ready = embedder is not None
        return {
            "status": "ok" if index_ready and embedder_ready else "degraded",
            "chroma_ready": index_ready,
            "embedder_ready": embedder_ready,
        }

    @app.post("/api/sessions")
    def create_session(room: Room) -> dict[str, str]:
        return {"session_id": store.create(room)}

    @app.get("/api/sessions/{session_id}/plan")
    def get_plan(session_id: str) -> dict[str, Any]:
        try:
            session = store.get(session_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="session not found") from exc
        return {
            "plan": None if session.plan is None else session.plan.model_dump(),
            "products": session.products,
            "metrics": None if session.metrics is None else session.metrics.model_dump(),
        }

    @app.post("/api/sessions/{session_id}/chat")
    async def chat(session_id: str, body: ChatRequest) -> StreamingResponse:
        try:
            session = store.get(session_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="session not found") from exc

        history = [message.model_dump() for message in session.messages]
        selected = list(session.products)
        result = await asyncio.to_thread(
            graph.invoke,
            {
                "room": session.room.model_dump(),
                "message": body.message,
                "history": history,
                "selected_products": selected,
            },
        )
        response_text = str(result["response_text"])
        plan = DesignPlan.model_validate(result["plan"])
        metrics = UsageMetrics.model_validate(result["metrics"])
        products = list(result["products"])
        store.append_message(session_id, ChatMessage(role="user", content=body.message))
        store.append_message(session_id, ChatMessage(role="assistant", content=response_text))
        store.set_plan(session_id, plan, products=products, metrics=metrics)
        payload = {
            "plan": plan.model_dump(),
            "products": products,
            "metrics": metrics.model_dump(),
        }

        async def events():
            for chunk in iter_token_chunks(response_text):
                yield _sse("token", {"text": chunk})
            yield _sse("plan", payload)
            yield _sse("done", {})

        return StreamingResponse(events(), media_type="text/event-stream")

    return app


app = create_app()

"""In-memory design sessions and the LangGraph state channels."""

from __future__ import annotations

import threading
import uuid
from typing import TypedDict

from pydantic import BaseModel, Field

from app.metrics import UsageMetrics
from app.schemas import ChatMessage, DesignPlan, Room


class Session(BaseModel):
    """One shopper's room, transcript, and latest grounded plan."""

    session_id: str
    room: Room
    messages: list[ChatMessage] = Field(default_factory=list)
    plan: DesignPlan | None = None
    products: list[dict] = Field(default_factory=list)
    metrics: UsageMetrics | None = None


class SessionStore:
    """Process-local sessions. ``get`` returns a deep copy."""

    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()

    def create(self, room: Room) -> str:
        session_id = uuid.uuid4().hex
        with self._lock:
            self._sessions[session_id] = Session(session_id=session_id, room=room)
        return session_id

    def get(self, session_id: str) -> Session:
        with self._lock:
            try:
                session = self._sessions[session_id]
            except KeyError as exc:
                raise KeyError(session_id) from exc
            return session.model_copy(deep=True)

    def append_message(self, session_id: str, message: ChatMessage) -> None:
        with self._lock:
            self._sessions[session_id].messages.append(message)

    def set_plan(
        self,
        session_id: str,
        plan: DesignPlan,
        products: list[dict] | None = None,
        metrics: UsageMetrics | None = None,
    ) -> None:
        with self._lock:
            session = self._sessions[session_id]
            session.plan = plan
            if products is not None:
                session.products = list(products)
            if metrics is not None:
                session.metrics = metrics


class AgentState(TypedDict, total=False):
    """Channels for parse → retrieve → constrain → plan → respond."""

    room: dict
    message: str
    history: list[dict]
    selected_products: list[dict]
    remaining_budget: float
    retrieval_query: str
    filters: dict
    candidates: list[dict]
    accepted: list[dict]
    rejected_notes: list[str]
    plan: dict
    products: list[dict]
    response_text: str
    metrics: dict
    started_at: float
    prompt_tokens: int
    completion_tokens: int

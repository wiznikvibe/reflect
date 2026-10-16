# Nestingale Design Planning RAG Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship an end-to-end Nestingale Design Planning RAG PoC: living-room catalog ingest → hybrid retrieval → constraint-checked LangGraph plan → FastAPI SSE → React UI → eval harness.

**Architecture:** Thin LangGraph (`parse → retrieve → constrain → plan → respond`) over Chroma + local EmbeddingGemma; Gemini Flash for structured plans; deterministic Python constraints; React SPA for chat/plan/products/metrics.

**Tech Stack:** Python 3.11+, FastAPI, LangGraph, Chroma, `google/embeddinggemma-300m`, Google Gemini Flash, Pydantic, React + Vite, pytest.

## Global Constraints

- Living room only; catalog size 50–70 SKUs across sofas, chairs, coffee/side tables, lighting, rugs.
- Embeddings: Hugging Face `google/embeddinggemma-300m` (local); planning LLM: Google Gemini Flash.
- Vector store: Chroma persisted at `backend/.chroma/`; rebuild via `python -m app.ingest --rebuild`.
- Agent graph is fixed (not free-form ReAct): parse → retrieve → constrain → plan → respond.
- Constraints are deterministic Python (fit, budget, style); LLM must not invent product IDs.
- Retrieval `top_k=12` before constraints; plan on ≤8 candidates.
- API: FastAPI with SSE chat; endpoints `/api/sessions`, `/api/sessions/{id}/chat`, `/api/sessions/{id}/plan`, `/api/health`.
- UI: React (Vite) SPA; no Three.js / no 3D / no floor-plan geometry.
- No multimodal reference images on day 1; eval set of 5–10 queries required.
- No auth, K8s, fine-tuning, or production monitoring.
- Work from repo root `/home/kai/repo/reflect` on branch `feature/nestingale-design-rag-poc`.
- Follow TDD for backend logic tasks; commit after each task.

## File structure

```
data/catalog/living_room.json
data/rooms/sample_living_room.json
data/evals/queries.json
backend/
  requirements.txt
  app/__init__.py
  app/schemas.py
  app/ingest.py
  app/embeddings.py
  app/retrieval.py
  app/constraints.py
  app/metrics.py
  app/agent/state.py
  app/agent/graph.py
  app/main.py
  scripts/run_eval.py
  tests/...
frontend/
  package.json
  vite.config.ts
  src/App.tsx
  src/api.ts
  src/components/{RoomPanel,Chat,PlanView,ProductList,MetricsBar}.tsx
  src/styles.css
.env.example
README.md
```

---

### Task 1: Schemas, catalog, and sample room

**Files:**
- Create: `backend/requirements.txt`
- Create: `backend/app/__init__.py`
- Create: `backend/app/schemas.py`
- Create: `backend/tests/test_schemas.py`
- Create: `data/catalog/living_room.json`
- Create: `data/rooms/sample_living_room.json`
- Create: `.env.example`

**Interfaces:**
- Consumes: none
- Produces: Pydantic models `Room`, `Opening`, `Dimensions`, `Product`, `ProductDimensions`, `DesignPlanItem`, `DesignPlan`, `ChatMessage`; catalog with 50–70 living-room products; sample room matching design-spec JSON shape

- [ ] **Step 1: Write failing schema tests** in `backend/tests/test_schemas.py` covering Room parse from sample JSON, Product required fields, DesignPlan item grounding fields.
- [ ] **Step 2: Run tests — expect FAIL** (`pytest backend/tests/test_schemas.py -v`).
- [ ] **Step 3: Implement `schemas.py` + `requirements.txt` (pydantic, pytest, fastapi, uvicorn, chromadb, sentence-transformers/transformers, torch as needed, langgraph, langchain-google-genai, python-dotenv, httpx) + `.env.example` with `GOOGLE_API_KEY`, `GEMINI_MODEL=gemini-2.0-flash`, embedding model id, cost rates.
- [ ] **Step 4: Author `data/catalog/living_room.json` (50–70 realistic SKUs with all product fields) and `data/rooms/sample_living_room.json`.**
- [ ] **Step 5: Run tests — expect PASS.**
- [ ] **Step 6: Commit** with message `feat: add schemas, living-room catalog, and sample room`.

---

### Task 2: Ingestion pipeline (EmbeddingGemma → Chroma)

**Files:**
- Create: `backend/app/embeddings.py`
- Create: `backend/app/ingest.py`
- Create: `backend/tests/test_ingest.py`

**Interfaces:**
- Consumes: `Product` schema; `data/catalog/living_room.json`
- Produces:
  - `get_embedder() -> Embedder` with `embed_documents(texts: list[str]) -> list[list[float]]` and `embed_query(text: str) -> list[float]`
  - `ingest_catalog(catalog_path, chroma_dir, rebuild: bool=False) -> int` (count upserted)
  - `get_collection(chroma_dir)` for later retrieval
  - Embed text format: `name | category | styles | material | description`
  - Metadata: `id`, `category`, `price_usd`, `width_cm`, `depth_cm`, `height_cm`, `style_tags` (stringified), `material`, `name`
  - CLI: `python -m app.ingest --rebuild` from `backend/`

- [ ] **Step 1: Write failing tests** for clean/normalize + ingest with a tiny fixture catalog (3 items) into a temp chroma dir; assert count and metadata keys. Mock or skip heavy model download in unit tests by injecting a FakeEmbedder; add one optional integration mark if needed.
- [ ] **Step 2: Run tests — expect FAIL.**
- [ ] **Step 3: Implement embeddings wrapper (model `google/embeddinggemma-300m`) and ingest module with persist + `--rebuild`.**
- [ ] **Step 4: Run tests — expect PASS.**
- [ ] **Step 5: Commit** `feat: ingest living-room catalog into Chroma with EmbeddingGemma`.

---

### Task 3: Hybrid retrieval and constraint engine

**Files:**
- Create: `backend/app/retrieval.py`
- Create: `backend/app/constraints.py`
- Create: `backend/tests/test_retrieval.py`
- Create: `backend/tests/test_constraints.py`

**Interfaces:**
- Consumes: Chroma collection; `Room`; product metadata
- Produces:
  - `retrieve(query: str, room: Room, filters: RetrievalFilters | None, top_k: int = 12) -> list[ProductCandidate]`
  - `RetrievalFilters`: optional `categories`, `max_price_usd`, `max_width_cm`
  - `usable_wall_lengths_m(room: Room) -> dict[str, float]`
  - `apply_constraints(candidates, room, selected_so_far, remaining_budget) -> ConstraintResult` with `accepted`, `rejected` (each with reason), style overlap score
  - Fit rule: for sofas/chairs, `width_cm/100 <= max(usable_wall_lengths)` (simple heuristic)
  - Budget: `price_usd <= remaining_budget`
  - Style: overlap between product tags and tokenized `style_preference`

- [ ] **Step 1: Write failing tests** for wall usable length with openings, fit reject/accept, budget reject, style overlap, retrieve filter wiring with FakeEmbedder + temp chroma.
- [ ] **Step 2: Run — expect FAIL.**
- [ ] **Step 3: Implement retrieval + constraints.**
- [ ] **Step 4: Run — expect PASS.**
- [ ] **Step 5: Commit** `feat: add hybrid retrieval and deterministic constraints`.

---

### Task 4: LangGraph agent + FastAPI SSE

**Files:**
- Create: `backend/app/agent/__init__.py`
- Create: `backend/app/agent/state.py`
- Create: `backend/app/agent/graph.py`
- Create: `backend/app/metrics.py`
- Create: `backend/app/main.py`
- Create: `backend/tests/test_api.py`
- Create: `backend/tests/test_agent_graph.py`

**Interfaces:**
- Consumes: retrieval, constraints, schemas, Gemini via env `GOOGLE_API_KEY`
- Produces:
  - LangGraph compiled app with nodes parse → retrieve → constrain → plan → respond
  - In-memory `SessionStore`: `create(room) -> session_id`, `get`, `append_message`, `set_plan`
  - FastAPI app with CORS for Vite (`http://localhost:5173`)
  - `POST /api/sessions` body `Room` → `{session_id}`
  - `POST /api/sessions/{id}/chat` body `{message: str}` → SSE stream: `event: token` then `event: plan` (DesignPlan + products + metrics) then `event: done`
  - `GET /api/sessions/{id}/plan`
  - `GET /api/health` → `{status, chroma_ready, embedder_ready}`
  - Metrics: latency_ms, prompt/completion tokens, approx_cost_usd
  - Plan prompt forbids inventing SKUs; only use constrained candidate IDs
  - Tests may mock Gemini; health/session create should work without live LLM where possible

- [ ] **Step 1: Write failing API/agent tests** (httpx AsyncClient) with mocked LLM + FakeEmbedder.
- [ ] **Step 2: Run — expect FAIL.**
- [ ] **Step 3: Implement graph, metrics, FastAPI.**
- [ ] **Step 4: Run — expect PASS.**
- [ ] **Step 5: Commit** `feat: LangGraph design agent with FastAPI SSE sessions`.

---

### Task 5: React SPA UI

**Files:**
- Create: `frontend/package.json`, `vite.config.ts`, `tsconfig.json`, `index.html`
- Create: `frontend/src/main.tsx`, `frontend/src/App.tsx`, `frontend/src/api.ts`
- Create: `frontend/src/components/RoomPanel.tsx`, `Chat.tsx`, `PlanView.tsx`, `ProductList.tsx`, `MetricsBar.tsx`
- Create: `frontend/src/styles.css`

**Interfaces:**
- Consumes: FastAPI endpoints above (base URL `http://localhost:8000`)
- Produces: Responsive UI — room summary + editable JSON/budget, streaming chat, plan cards, ranked products with why_it_fits, metrics strip (latency, tokens, cost)
- Visual: warm neutrals + one accent; expressive fonts (not Inter/Roboto/Arial/system); subtle gradient/texture; no purple-glow cliché; no Three.js

- [ ] **Step 1: Scaffold Vite React+TS app and wire API client with SSE parser.**
- [ ] **Step 2: Build layout components and styles.**
- [ ] **Step 3: Manual smoke: load sample room, create session, send message (mock backend OK if documented).**
- [ ] **Step 4: Commit** `feat: add React demo UI for design planning chat`.

---

### Task 6: Eval harness + README

**Files:**
- Create: `data/evals/queries.json` (5–10 cases)
- Create: `backend/scripts/run_eval.py`
- Create: `backend/tests/test_eval_fixtures.py`
- Create: `README.md`

**Interfaces:**
- Consumes: retrieval, constraints, agent (LLM mockable for CI)
- Produces: CLI report — retrieval hit rate, constraint-violation rate, p50/p95 latency, mean approx cost
- README: architecture, setup (API key, ingest, backend, frontend), real vs mocked table, next production steps, latency notes

- [ ] **Step 1: Write eval fixtures + test that fixtures validate against schemas.**
- [ ] **Step 2: Implement `run_eval.py`.**
- [ ] **Step 3: Write README.**
- [ ] **Step 4: Commit** `docs: add eval harness and README for Nestingale RAG PoC`.

---

## Done when

- Catalog ingested; health reports ready
- Multi-turn chat returns grounded plans under budget/fit constraints
- React UI demonstrates room → chat → plan → products → metrics
- Eval script runs on 5–10 queries and prints metrics
- README documents real vs mocked and next steps

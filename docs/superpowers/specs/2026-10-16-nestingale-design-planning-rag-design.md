# Nestingale Design Planning RAG Agent — Design Spec

**Date:** 2026-10-16  
**Status:** Approved for implementation  
**Approach:** Thin LangGraph + FastAPI + React (Approach 1)

## Goal

Build a focused Proof of Concept that consumes structured room scan data (mock Nestingale output) plus user preferences, retrieves shoppable living-room products via hybrid RAG, and produces a constraint-checked, explainable design plan through multi-turn chat — with evaluation hooks for retrieval quality, constraint adherence, latency, and approximate token cost.

## Non-goals

- 3D rendering, Three.js, floor-plan geometry, ControlNet, or room image generation
- Production auth, Kubernetes, or complex monitoring
- Fine-tuning
- Multimodal reference-image style extraction (deferred)
- Rebuilding Nestingale scanning or walkable scenes

## Locked decisions

| Decision | Choice |
|----------|--------|
| Scope | Living room only, ~50–70 SKUs |
| Categories | sofas, chairs, coffee/side tables, lighting, rugs |
| Embeddings | Local Hugging Face `google/embeddinggemma-300m` |
| Planning LLM | Google Gemini Flash |
| Vector store | Chroma (persisted under `backend/.chroma/`) |
| Agent | Thin LangGraph: parse → retrieve → constrain → plan → respond |
| API | FastAPI with SSE streaming |
| UI | React (Vite) SPA — stylish, responsive, low latency |
| Eval | 5–10 canned queries; hit rate + constraint-violation rate |
| Demo posture | Product-first with credible eng/eval hooks |

## Architecture

```
Ingestion:  catalog JSON → clean/normalize → EmbeddingGemma → Chroma (+ metadata)
Online:     React → FastAPI SSE → LangGraph → retrieve/constrain/plan → SSE plan+metrics
```

### Components

1. **Ingestion pipeline** — load catalog, normalize fields, embed `name | category | styles | material | description`, upsert to Chroma with metadata filters.
2. **Hybrid retrieval** — semantic search + metadata filters (category, max price, max width).
3. **Constraint engine** — deterministic Python (fit, budget, style overlap); LLM explains, does not invent fit math.
4. **LangGraph agent** — fixed graph nodes; grounded plan from constrained candidate IDs only.
5. **Session store** — in-memory multi-turn state.
6. **React UI** — room summary, streaming chat, plan + products, metrics strip.
7. **Eval harness** — offline script over fixtures.

## Data contracts

### Room input (mock scan)

```json
{
  "room_type": "living_room",
  "dimensions": {"length_m": 5.2, "width_m": 4.1, "height_m": 2.7},
  "openings": [{"type": "window", "wall": "north", "width_m": 1.8}],
  "existing_furniture": [],
  "style_preference": "modern scandinavian",
  "budget_usd": 2500,
  "user_notes": "Need a sofa that faces the window and a coffee table"
}
```

### Product

- `id`, `name`, `category`, `style_tags[]`
- `dimensions`: `{width_cm, depth_cm, height_cm}`
- `price_usd`, `material`, `description`

### Design plan (structured)

- `summary`, `style_direction`
- `items[]`: `{product_id, role, placement_note, why_it_fits}`
- `total_usd`, `budget_remaining_usd`
- `constraint_notes[]`
- `alternatives[]` (optional)

## Agent graph

1. **parse** — merge room + message + history into retrieval/plan brief.
2. **retrieve** — EmbeddingGemma query; `top_k=12`; filters for category/price/width.
3. **constrain** — fit (product width vs usable wall), budget (running total), style tag overlap; keep ≤8 candidates.
4. **plan** — Gemini Flash structured output; forbid inventing SKUs.
5. **respond** — stream text + final plan JSON + metrics.

## API

- `POST /api/sessions` — create session with room JSON
- `POST /api/sessions/{id}/chat` — SSE: token deltas, then `event: plan`
- `GET /api/sessions/{id}/plan` — current plan
- `GET /api/health` — index/model ready
- Dev: ingest rebuild via `python -m app.ingest --rebuild`

## Latency strategy

- Local embeddings; warm model + Chroma on startup
- Gemini Flash + SSE streaming
- Bounded `top_k` / candidate set
- Persist Chroma so restarts skip re-embed

## Evaluation

`data/evals/queries.json` (5–10 cases) with `relevant_product_ids` and `must_respect` (budget/fit).

Metrics: retrieval hit rate, constraint-violation rate, p50/p95 latency, mean approx cost.

## Real vs mocked

| Real | Mocked / deferred |
|------|-------------------|
| Catalog ingest, embeddings, Chroma, hybrid retrieve | Nestingale live scan API |
| Deterministic size/budget/style checks | Full spatial collision / layout solver |
| LangGraph + Gemini plan + multi-turn | Auth, prod monitoring, multimodal refs |
| React demo UI + eval script | 3D walkthrough / rendering |

## Next production steps

Wire Nestingale room API; Qdrant/pgvector; stronger layout constraints; product feed sync; multimodal style refs; auth and observability.

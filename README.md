# Nestingale Design Planning RAG

Proof of concept for a living-room design planner. A mock room scan plus a shopper message goes through hybrid retrieval, deterministic fit and budget checks, and a grounded Gemini plan. The React app shows the room, the chat, the plan, the products, and per-turn metrics.

`GEMINI_MODEL=gemini-2.5-flash`

## Architecture

```
Ingestion:  catalog JSON → clean/normalize → EmbeddingGemma → Chroma (+ metadata)
Online:     React → FastAPI SSE → LangGraph → retrieve/constrain/plan → SSE plan+metrics
```

The agent graph is fixed: **parse → retrieve → constrain → plan → respond**.

| Piece | Role |
|---|---|
| `data/catalog/living_room.json` | Living-room SKUs: sofas, chairs, coffee tables, side tables, lighting, rugs |
| `backend/app/ingest.py` | Normalize, embed `name \| category \| styles \| material \| description`, upsert Chroma |
| `backend/app/retrieval.py` | EmbeddingGemma query, `top_k=12`, filters for category, max price, max width |
| `backend/app/constraints.py` | Deterministic fit, budget, and style-overlap checks |
| `backend/app/agent/graph.py` | LangGraph planner. At most 8 candidates, and only those product ids |
| `backend/app/main.py` | FastAPI sessions and SSE chat |
| `frontend/` | Vite + React demo |
| `backend/scripts/run_eval.py` | Offline hit rate, constraint-violation rate, latency, approximate cost |

Chroma is persisted at `backend/.chroma/`. Embeddings are local (`google/embeddinggemma-300m`). The planner is Gemini Flash via `GOOGLE_API_KEY`.

## Setup

Python 3.11+. From the repo root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
```

Copy the template and set the API key in the ignored env file:

```bash
cp .env.example .env
```

Put `GOOGLE_API_KEY` in `.env`. Leave `.env.example` with a blank key. `.env` is gitignored.

```
GOOGLE_API_KEY=
GEMINI_MODEL=gemini-2.5-flash
EMBEDDING_MODEL=google/embeddinggemma-300m
```

Ingest the catalog. The first run downloads EmbeddingGemma.

```bash
cd backend
python -m app.ingest --rebuild
```

Start the API:

```bash
uvicorn app.main:app --reload --port 8000
```

`GET http://localhost:8000/api/health` reports `chroma_ready` and `embedder_ready` once the index and embedder are available.

In another shell, start the UI:

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`. The sample room loads from `data/rooms/sample_living_room.json`. CORS allows that Vite origin.

Backend tests, from `backend/`:

```bash
python -m pytest
```

## Eval

Fixtures live in `data/evals/queries.json` (8 cases). Each case has a room fixture path, a user query, `relevant_product_ids`, and `must_respect` (`budget`, `fit`, or both). Extra rooms for a $100 budget, a 1.6 m wall, and a mid-century brief are under `data/evals/rooms/`.

From `backend/`:

```bash
python scripts/run_eval.py --mock
python scripts/run_eval.py --mock --retrieval-only
python scripts/run_eval.py
```

| Flag | What runs |
|---|---|
| `--mock` | `FakeEmbedder`, a temporary Chroma index, and a scripted planner. No API key and no model download. This is the CI path. |
| `--mock --retrieval-only` | Same index, retrieval and constraints only. Approximate cost is 0. |
| (none) | Persisted `backend/.chroma/` and live Gemini. Requires `GOOGLE_API_KEY` in `.env` and a prior ingest. |

The report prints:

- **Retrieval hit rate** — fraction of cases where at least one `relevant_product_ids` is in the top 12 after the agent's metadata filters.
- **Constraint-violation rate** — fraction of cases whose plan breaks a `must_respect` rule. `budget` means `total_usd` is above the room budget. `fit` means a sofa or chair is wider than the longest usable wall. `--retrieval-only` applies those checks to accepted candidates.
- **Latency p50 / p95** — milliseconds per case. The full path uses the graph's own latency. Retrieval-only times retrieve plus constrain.
- **Mean approx cost** — average estimated Gemini spend. Rates default to Gemini 2.5 Flash text: $0.30 / 1M input tokens and $2.50 / 1M output tokens. The scripted planner reports 120 prompt tokens and 30 completion tokens.

`--mock` uses `FakeEmbedder`, which hashes text, so rank inside a filter is not a semantic score. Several fixtures filter to fewer than 12 SKUs, and a mock hit rate can still be 1. Run `python scripts/run_eval.py` after ingest when you want EmbeddingGemma ranking.

## Real vs mocked

| Real | Mocked / deferred |
|---|---|
| Catalog ingest, embeddings, Chroma, hybrid retrieve | Nestingale live scan API |
| Deterministic size, budget, and style checks | Full spatial collision / layout solver |
| LangGraph + Gemini plan + multi-turn chat | Auth, production monitoring, multimodal reference images |
| React demo UI + eval script | 3D walkthrough / rendering |
| Live eval (`python scripts/run_eval.py`) | `--mock` eval: `FakeEmbedder` + scripted planner |

Room JSON in `data/rooms/` and `data/evals/rooms/` stands in for a Nestingale scan.

## Latency

- Embeddings stay on the machine. The first call loads EmbeddingGemma; later calls reuse it. Persisted Chroma means a restart skips re-embedding.
- The planner is `GEMINI_MODEL=gemini-2.5-flash` with `thinking_budget=0`.
- Retrieval is capped at `top_k=12`. The planner sees at most 8 candidates.
- The API times the whole graph, then streams the finished reply as 48-character SSE chunks. Time to the first token event is the graph time.
- Eval p50 and p95 describe this 8-case fixture set.

## Next production steps

Wire the Nestingale room API. Move the index to Qdrant or pgvector. Tighten layout constraints beyond wall length. Sync the product feed. Add multimodal style references. Add auth and observability.

"""Hybrid retrieval: semantic search plus Chroma metadata filters.

Chroma must be queried with ``query_embeddings`` from our embedder. Passing
``query_texts`` would run Chroma's default MiniLM model in the wrong space.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from app.embeddings import Embedder, get_embedder
from app.ingest import default_chroma_dir, get_collection, normalize_text, parse_style_tags
from app.schemas import Category, Room


class RetrievalFilters(BaseModel):
    """Optional metadata filters pushed into the Chroma ``where`` clause.

    An empty model, or ``None``, does not filter. An empty ``categories`` list
    does not filter either. Category values are the catalog literals
    (``sofas``, ``chairs``, and the rest), not singular names.
    """

    categories: list[Category] | None = None
    max_price_usd: float | None = Field(default=None, ge=0)
    max_width_cm: float | None = Field(default=None, gt=0)


class ProductCandidate(BaseModel):
    """One ranked SKU, before deterministic constraint checks."""

    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    category: Category
    style_tags: list[str] = Field(min_length=1)
    price_usd: float = Field(gt=0)
    width_cm: float = Field(gt=0)
    depth_cm: float = Field(gt=0)
    height_cm: float = Field(gt=0)
    material: str = Field(min_length=1)
    distance: float
    score: float
    document: str = ""


def retrieval_query_text(query: str, room: Room) -> str:
    """User text plus the room style preference.

    Metadata filters are the structured half of hybrid search. The preference
    is appended so the embedding leans toward that style. ``user_notes`` stay
    out of the vector string; the caller already put what they want in ``query``.
    """

    return f"{normalize_text(query)} | {normalize_text(room.style_preference)}"


def _where(filters: RetrievalFilters | None) -> dict[str, Any] | None:
    if filters is None:
        return None
    clauses: list[dict[str, Any]] = []
    if filters.categories:
        clauses.append({"category": {"$in": list(filters.categories)}})
    if filters.max_price_usd is not None:
        clauses.append({"price_usd": {"$lte": float(filters.max_price_usd)}})
    if filters.max_width_cm is not None:
        clauses.append({"width_cm": {"$lte": float(filters.max_width_cm)}})
    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def _candidate(
    item_id: str,
    metadata: dict[str, Any],
    document: str | None,
    distance: float | None,
) -> ProductCandidate:
    raw_tags = metadata.get("style_tags", "[]")
    if isinstance(raw_tags, str):
        style_tags = parse_style_tags(raw_tags)
    else:
        style_tags = [str(tag) for tag in raw_tags]
    dist = 0.0 if distance is None else float(distance)
    return ProductCandidate(
        id=str(metadata.get("id") or item_id),
        name=str(metadata["name"]),
        category=metadata["category"],
        style_tags=style_tags,
        price_usd=float(metadata["price_usd"]),
        width_cm=float(metadata["width_cm"]),
        depth_cm=float(metadata["depth_cm"]),
        height_cm=float(metadata["height_cm"]),
        material=str(metadata["material"]),
        distance=dist,
        score=1.0 - dist,
        document=document or "",
    )


def retrieve(
    query: str,
    room: Room,
    filters: RetrievalFilters | None = None,
    top_k: int = 12,
    *,
    chroma_dir: str | Path | None = None,
    embedder: Embedder | None = None,
    collection: Any | None = None,
) -> list[ProductCandidate]:
    """Return up to ``top_k`` products nearest the query in embedding space.

    ``embedder`` and ``chroma_dir`` are keyword-only so the brief's positional
    signature stays ``(query, room, filters, top_k)``. ``collection`` is a test
    seam. When ``embedder`` is omitted, ``get_embedder()`` is used.
    """

    if embedder is None:
        embedder = get_embedder()
    if collection is None:
        if chroma_dir is None:
            chroma_dir = default_chroma_dir()
        collection = get_collection(chroma_dir)

    vector = embedder.embed_query(retrieval_query_text(query, room))
    kwargs: dict[str, Any] = {
        "query_embeddings": [vector],
        "n_results": top_k,
        "include": ["metadatas", "documents", "distances"],
    }
    where = _where(filters)
    if where is not None:
        kwargs["where"] = where
    found = collection.query(**kwargs)

    ids = (found.get("ids") or [[]])[0] or []
    metadatas = (found.get("metadatas") or [[]])[0] or []
    documents = (found.get("documents") or [[]])[0] or []
    distances = (found.get("distances") or [[]])[0] or []
    candidates: list[ProductCandidate] = []
    for index, item_id in enumerate(ids):
        metadata = metadatas[index] if index < len(metadatas) and metadatas[index] else {}
        document = documents[index] if index < len(documents) else ""
        distance = distances[index] if index < len(distances) else None
        candidates.append(_candidate(str(item_id), metadata, document, distance))
    return candidates

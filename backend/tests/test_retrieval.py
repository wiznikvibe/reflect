"""Hybrid retrieval: embedder vectors plus Chroma metadata filters."""

import inspect
import json
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT))

from app.embeddings import FakeEmbedder  # noqa: E402
from app.ingest import embed_text, get_collection, ingest_catalog, normalize_product  # noqa: E402
from app.retrieval import (  # noqa: E402
    RetrievalFilters,
    retrieval_query_text,
    retrieve,
)
from app.schemas import Room  # noqa: E402

FIXTURE = [
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
        "id": "chair-lumen",
        "name": "Lumen Lounge Chair",
        "category": "chairs",
        "style_tags": ["modern"],
        "dimensions": {"width_cm": 78, "depth_cm": 82, "height_cm": 76},
        "price_usd": 420.5,
        "material": "leather",
        "description": "A compact lounge chair.",
    },
    {
        "id": "rug-fjord",
        "name": "Fjord Wool Rug",
        "category": "rugs",
        "style_tags": ["scandinavian", "minimalist"],
        "dimensions": {"width_cm": 160, "depth_cm": 230, "height_cm": 1},
        "price_usd": 240,
        "material": "wool",
        "description": "Flatweave wool rug.",
    },
]


class _SpyCollection:
    def __init__(self, inner) -> None:
        self.inner = inner
        self.calls: list[dict] = []

    def query(self, *args, **kwargs):
        self.calls.append({"args": args, "kwargs": kwargs})
        return self.inner.query(*args, **kwargs)


def _room() -> Room:
    return Room.model_validate(
        {
            "room_type": "living_room",
            "dimensions": {"length_m": 5.0, "width_m": 4.0, "height_m": 2.5},
            "openings": [],
            "existing_furniture": [],
            "style_preference": "modern scandinavian",
            "budget_usd": 2500,
            "user_notes": "Need a sofa that faces the window",
        }
    )


def _ingest(tmp_path: Path):
    catalog = tmp_path / "catalog.json"
    catalog.write_text(json.dumps(FIXTURE), encoding="utf-8")
    chroma_dir = tmp_path / "chroma"
    embedder = FakeEmbedder()
    ingest_catalog(catalog, chroma_dir, rebuild=True, embedder=embedder)
    return chroma_dir, embedder


def test_top_k_defaults_to_12():
    assert inspect.signature(retrieve).parameters["top_k"].default == 12


def test_retrieval_filters_use_catalog_categories():
    filters = RetrievalFilters(
        categories=["sofas", "chairs"],
        max_price_usd=500,
        max_width_cm=90,
    )
    assert filters.categories == ["sofas", "chairs"]
    assert filters.max_price_usd == 500
    assert filters.max_width_cm == 90

    with pytest.raises(ValidationError):
        RetrievalFilters(categories=["sofa"])


def test_retrieval_query_appends_style_preference_only():
    room = _room()
    text = retrieval_query_text("  lounge\nchair  ", room)
    assert text == "lounge chair | modern scandinavian"
    assert "window" not in text


def test_retrieve_queries_with_embeddings_and_metadata_filters(tmp_path):
    chroma_dir, embedder = _ingest(tmp_path)
    room = _room()
    spy = _SpyCollection(get_collection(chroma_dir))
    filters = RetrievalFilters(
        categories=["sofas", "rugs"],
        max_price_usd=1000,
        max_width_cm=220,
    )

    results = retrieve(
        "wool rug",
        room,
        filters,
        top_k=2,
        embedder=embedder,
        collection=spy,
    )

    assert len(spy.calls) == 1
    kwargs = spy.calls[0]["kwargs"]
    assert spy.calls[0]["args"] == ()
    assert "query_texts" not in kwargs
    expected = embedder.embed_query("wool rug | modern scandinavian")
    assert kwargs["query_embeddings"] == [expected]
    assert kwargs["n_results"] == 2
    assert kwargs["where"] == {
        "$and": [
            {"category": {"$in": ["sofas", "rugs"]}},
            {"price_usd": {"$lte": 1000.0}},
            {"width_cm": {"$lte": 220.0}},
        ]
    }
    assert {item.id for item in results} == {"sofa-oslo", "rug-fjord"}
    assert all(item.category in {"sofas", "rugs"} for item in results)


def test_retrieve_omits_where_when_filters_are_empty(tmp_path):
    chroma_dir, embedder = _ingest(tmp_path)
    spy = _SpyCollection(get_collection(chroma_dir))

    retrieve("sofa", _room(), None, embedder=embedder, collection=spy)
    retrieve("sofa", _room(), RetrievalFilters(), embedder=embedder, collection=spy)

    assert "where" not in spy.calls[0]["kwargs"]
    assert "where" not in spy.calls[1]["kwargs"]
    assert spy.calls[0]["kwargs"]["n_results"] == 12
    assert "query_texts" not in spy.calls[0]["kwargs"]


def test_retrieve_filter_wiring_against_temp_chroma(tmp_path):
    chroma_dir, embedder = _ingest(tmp_path)
    room = _room()

    sofas = retrieve(
        "sofa",
        room,
        RetrievalFilters(categories=["sofas"]),
        chroma_dir=chroma_dir,
        embedder=embedder,
    )
    affordable = retrieve(
        "rug",
        room,
        RetrievalFilters(max_price_usd=300),
        chroma_dir=chroma_dir,
        embedder=embedder,
    )
    narrow = retrieve(
        "chair",
        room,
        RetrievalFilters(max_width_cm=100),
        chroma_dir=chroma_dir,
        embedder=embedder,
    )
    none = retrieve(
        "sofa",
        room,
        RetrievalFilters(categories=["sofas"], max_price_usd=100),
        chroma_dir=chroma_dir,
        embedder=embedder,
    )

    assert [item.id for item in sofas] == ["sofa-oslo"]
    assert sofas[0].name == "Oslo Three-Seat Sofa"
    assert sofas[0].style_tags == ["modern", "scandinavian"]
    assert sofas[0].material == "oatmeal linen"
    assert sofas[0].price_usd == pytest.approx(899)
    assert sofas[0].width_cm == pytest.approx(210)
    assert sofas[0].depth_cm == pytest.approx(88)
    assert sofas[0].height_cm == pytest.approx(80)
    assert sofas[0].document == embed_text(normalize_product(FIXTURE[0]))
    assert [item.id for item in affordable] == ["rug-fjord"]
    assert [item.id for item in narrow] == ["chair-lumen"]
    assert narrow[0].style_tags == ["modern"]
    assert narrow[0].price_usd == pytest.approx(420.5)
    assert none == []


def test_retrieve_preserves_chroma_rank_order(tmp_path):
    chroma_dir, embedder = _ingest(tmp_path)
    room = _room()
    text = retrieval_query_text("scandinavian seating", room)
    collection = get_collection(chroma_dir)
    direct = collection.query(
        query_embeddings=[embedder.embed_query(text)],
        n_results=2,
        include=["metadatas", "documents", "distances"],
    )

    ranked = retrieve(
        "scandinavian seating",
        room,
        top_k=2,
        chroma_dir=chroma_dir,
        embedder=embedder,
    )

    assert [item.id for item in ranked] == direct["ids"][0]
    assert ranked[0].distance == pytest.approx(direct["distances"][0][0])
    assert ranked[0].score == pytest.approx(1.0 - direct["distances"][0][0])


def test_retrieve_uses_get_embedder_when_none_is_injected(tmp_path, monkeypatch):
    chroma_dir, embedder = _ingest(tmp_path)
    monkeypatch.setattr("app.retrieval.get_embedder", lambda: embedder)

    results = retrieve(
        "chair",
        _room(),
        RetrievalFilters(categories=["chairs"]),
        chroma_dir=chroma_dir,
    )

    assert [item.id for item in results] == ["chair-lumen"]

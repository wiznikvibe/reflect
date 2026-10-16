"""Ingest tests. Unit tests inject FakeEmbedder and never download EmbeddingGemma."""

import json
import os
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

BACKEND_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_ROOT.parent
sys.path.insert(0, str(BACKEND_ROOT))

from app.embeddings import (  # noqa: E402
    DEFAULT_MODEL_ID,
    DOCUMENT_PROMPT,
    QUERY_PROMPT,
    EmbeddingGemmaEmbedder,
    FakeEmbedder,
    get_embedder,
)
from app.ingest import (  # noqa: E402
    COLLECTION_NAME,
    default_catalog_path,
    default_chroma_dir,
    embed_text,
    get_collection,
    ingest_catalog,
    load_catalog,
    main,
    normalize_product,
    parse_style_tags,
    product_metadata,
    stringify_style_tags,
)

CATALOG_PATH = REPO_ROOT / "data" / "catalog" / "living_room.json"
METADATA_KEYS = (
    "id",
    "category",
    "price_usd",
    "width_cm",
    "depth_cm",
    "height_cm",
    "style_tags",
    "material",
    "name",
)

FIXTURE = [
    {
        "id": " sofa-oslo ",
        "name": "  Oslo   Three-Seat   Sofa ",
        "category": "sofas",
        "style_tags": [" Modern ", "modern", "scandinavian", " "],
        "dimensions": {"width_cm": 210, "depth_cm": 88, "height_cm": 80},
        "price_usd": 899,
        "material": "  oatmeal linen  ",
        "description": "Low-profile\nthree-seat   sofa.",
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

OSLO_EMBED_TEXT = (
    "Oslo Three-Seat Sofa | sofas | modern, scandinavian | "
    "oatmeal linen | Low-profile three-seat sofa."
)


def _write_catalog(path: Path, items: list[dict]) -> Path:
    path.write_text(json.dumps(items), encoding="utf-8")
    return path


def _rows(embeddings) -> list[list[float]]:
    return [[float(value) for value in row] for row in embeddings]


class _RecordingEmbedder:
    def __init__(self) -> None:
        self.inner = FakeEmbedder()
        self.batches: list[list[str]] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.batches.append(list(texts))
        return self.inner.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self.inner.embed_query(text)


class _StubModel:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def encode(self, sentences, **kwargs):
        self.calls.append({"sentences": sentences, "kwargs": kwargs})
        if not sentences:
            return []
        return [[float(index + 1), 0.25] for index, _ in enumerate(sentences)]


def test_normalize_product_cleans_text_and_style_tags():
    product = normalize_product(FIXTURE[0])

    assert product.id == "sofa-oslo"
    assert product.name == "Oslo Three-Seat Sofa"
    assert product.material == "oatmeal linen"
    assert product.description == "Low-profile three-seat sofa."
    assert product.style_tags == ["modern", "scandinavian"]
    assert product.category == "sofas"
    assert product.price_usd == 899
    assert product.dimensions.width_cm == 210


def test_normalize_product_rejects_blank_style_tags():
    raw = dict(FIXTURE[1])
    raw["style_tags"] = ["  "]
    with pytest.raises(ValidationError):
        normalize_product(raw)


def test_embed_text_uses_name_category_styles_material_description():
    product = normalize_product(FIXTURE[0])
    assert embed_text(product) == OSLO_EMBED_TEXT

    chair = normalize_product(FIXTURE[1])
    assert embed_text(chair) == (
        "Lumen Lounge Chair | chairs | modern | leather | A compact lounge chair."
    )


def test_product_metadata_is_flat_and_stringifies_style_tags():
    product = normalize_product(FIXTURE[0])
    metadata = product_metadata(product)

    assert tuple(metadata) == METADATA_KEYS
    assert metadata["id"] == "sofa-oslo"
    assert metadata["category"] == "sofas"
    assert metadata["name"] == "Oslo Three-Seat Sofa"
    assert metadata["material"] == "oatmeal linen"
    assert metadata["price_usd"] == 899.0
    assert metadata["width_cm"] == 210.0
    assert metadata["depth_cm"] == 88.0
    assert metadata["height_cm"] == 80.0
    assert isinstance(metadata["style_tags"], str)
    assert metadata["style_tags"] == stringify_style_tags(["modern", "scandinavian"])
    assert parse_style_tags(metadata["style_tags"]) == ["modern", "scandinavian"]
    for value in metadata.values():
        assert isinstance(value, (str, float))


def test_fake_embedder_is_deterministic_and_normalized():
    embedder = FakeEmbedder()
    sofa = embedder.embed_query("sofa")
    again = embedder.embed_query("sofa")
    chair = embedder.embed_query("chair")

    assert sofa == again
    assert sofa != chair
    assert len(sofa) == 32
    assert sum(value * value for value in sofa) == pytest.approx(1.0)
    assert embedder.embed_documents(["sofa", "chair"]) == [sofa, chair]
    assert embedder.embed_documents([]) == []


def test_gemma_wrapper_uses_retrieval_prompts_without_loading_a_real_model():
    model = _StubModel()
    embedder = EmbeddingGemmaEmbedder(model=model)
    assert embedder.model_id == DEFAULT_MODEL_ID
    assert embedder.is_loaded is True

    documents = embedder.embed_documents(["alpha", "beta"])
    query = embedder.embed_query("gamma")

    assert documents == [[1.0, 0.25], [2.0, 0.25]]
    assert query == [1.0, 0.25]
    assert embedder.embed_documents([]) == []
    assert model.calls[0]["sentences"] == ["alpha", "beta"]
    assert model.calls[0]["kwargs"]["prompt"] == DOCUMENT_PROMPT
    assert model.calls[0]["kwargs"]["normalize_embeddings"] is True
    assert model.calls[1]["sentences"] == ["gamma"]
    assert model.calls[1]["kwargs"]["prompt"] == QUERY_PROMPT
    assert DOCUMENT_PROMPT == "title: none | text: "
    assert QUERY_PROMPT == "task: search result | query: "
    assert len(model.calls) == 2


def test_get_embedder_is_lazy_and_does_not_construct_sentence_transformer(monkeypatch):
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
    import app.embeddings as embeddings

    embeddings._CACHED_EMBEDDER = None

    class BoomSentenceTransformer:
        def __init__(self, *args, **kwargs):
            raise AssertionError("SentenceTransformer must stay lazy until embed time")

    import types

    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = BoomSentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)

    embedder = get_embedder()
    assert isinstance(embedder, EmbeddingGemmaEmbedder)
    assert embedder.model_id == "google/embeddinggemma-300m"
    assert embedder.is_loaded is False
    assert get_embedder() is embedder


def test_model_id_can_be_overridden_from_the_environment(monkeypatch):
    monkeypatch.setenv("EMBEDDING_MODEL", "org/custom-embedder")
    embedder = EmbeddingGemmaEmbedder()
    assert embedder.model_id == "org/custom-embedder"
    assert embedder.is_loaded is False


def test_load_constructs_sentence_transformer_once(monkeypatch):
    created: list[str] = []

    class FakeSentenceTransformer:
        def __init__(self, model_id: str):
            created.append(model_id)

        def encode(self, sentences, **kwargs):
            return [[0.0, 1.0] for _ in sentences]

    import types

    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = FakeSentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)

    embedder = EmbeddingGemmaEmbedder()
    assert embedder.is_loaded is False
    first = embedder.embed_query("hello")
    second = embedder.embed_documents(["hello"])
    assert first == [0.0, 1.0]
    assert second == [[0.0, 1.0]]
    assert created == ["google/embeddinggemma-300m"]
    assert embedder.is_loaded is True


def test_living_room_catalog_normalizes_into_embed_text():
    products = load_catalog(CATALOG_PATH)
    assert 50 <= len(products) <= 70
    assert len({product.id for product in products}) == len(products)

    for product in products:
        styles = ", ".join(product.style_tags)
        assert embed_text(product) == (
            f"{product.name} | {product.category} | {styles} | "
            f"{product.material} | {product.description}"
        )
        metadata = product_metadata(product)
        assert tuple(metadata) == METADATA_KEYS
        assert parse_style_tags(metadata["style_tags"]) == product.style_tags


def test_load_catalog_rejects_bad_shapes(tmp_path):
    missing = tmp_path / "missing.json"
    with pytest.raises(FileNotFoundError):
        load_catalog(missing)

    not_a_list = tmp_path / "object.json"
    not_a_list.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON list"):
        load_catalog(not_a_list)

    duplicates = tmp_path / "dupes.json"
    _write_catalog(
        duplicates,
        [FIXTURE[1], {**FIXTURE[2], "id": "chair-lumen"}],
    )
    with pytest.raises(ValueError, match="duplicate"):
        load_catalog(duplicates)


def test_ingest_persists_count_metadata_and_vectors(tmp_path):
    catalog = _write_catalog(tmp_path / "catalog.json", FIXTURE)
    chroma_dir = tmp_path / "chroma"
    embedder = _RecordingEmbedder()

    count = ingest_catalog(catalog, chroma_dir, embedder=embedder)

    assert count == 3
    assert embedder.batches == [[
        OSLO_EMBED_TEXT,
        "Lumen Lounge Chair | chairs | modern | leather | A compact lounge chair.",
        "Fjord Wool Rug | rugs | scandinavian, minimalist | wool | Flatweave wool rug.",
    ]]

    collection = get_collection(chroma_dir)
    assert collection.name == COLLECTION_NAME
    assert collection.count() == 3

    stored = collection.get(include=["metadatas", "documents", "embeddings"])
    by_id = {item_id: index for index, item_id in enumerate(stored["ids"])}
    assert set(by_id) == {"sofa-oslo", "chair-lumen", "rug-fjord"}

    products = [normalize_product(item) for item in FIXTURE]
    texts = [embed_text(product) for product in products]
    expected_vectors = embedder.inner.embed_documents(texts)
    for index, product in enumerate(products):
        slot = by_id[product.id]
        metadata = stored["metadatas"][slot]
        assert set(metadata) == set(METADATA_KEYS)
        assert metadata["id"] == product.id
        assert metadata["category"] == product.category
        assert metadata["name"] == product.name
        assert metadata["material"] == product.material
        assert metadata["style_tags"] == stringify_style_tags(product.style_tags)
        assert float(metadata["price_usd"]) == pytest.approx(product.price_usd)
        assert float(metadata["width_cm"]) == pytest.approx(product.dimensions.width_cm)
        assert float(metadata["depth_cm"]) == pytest.approx(product.dimensions.depth_cm)
        assert float(metadata["height_cm"]) == pytest.approx(product.dimensions.height_cm)
        assert stored["documents"][slot] == texts[index]
        assert _rows([stored["embeddings"][slot]])[0] == pytest.approx(expected_vectors[index])

    sofas = collection.get(where={"category": "sofas"})
    assert sofas["ids"] == ["sofa-oslo"]
    affordable = collection.get(where={"price_usd": {"$lte": 300}})
    assert affordable["ids"] == ["rug-fjord"]
    narrow = collection.get(where={"width_cm": {"$lte": 100}})
    assert narrow["ids"] == ["chair-lumen"]

    nearest = collection.query(
        query_embeddings=[embedder.embed_query(OSLO_EMBED_TEXT)],
        n_results=1,
    )
    assert nearest["ids"] == [["sofa-oslo"]]


def test_upsert_updates_existing_rows_and_rebuild_drops_them(tmp_path):
    catalog = _write_catalog(tmp_path / "catalog.json", FIXTURE)
    chroma_dir = tmp_path / "chroma"
    embedder = FakeEmbedder()

    assert ingest_catalog(catalog, chroma_dir, rebuild=False, embedder=embedder) == 3

    updated_chair = dict(FIXTURE[1])
    updated_chair["price_usd"] = 111
    updated_chair["description"] = "Updated chair copy."
    _write_catalog(tmp_path / "one.json", [updated_chair])
    assert ingest_catalog(tmp_path / "one.json", chroma_dir, embedder=embedder) == 1
    assert get_collection(chroma_dir).count() == 3
    chair = get_collection(chroma_dir).get(ids=["chair-lumen"], include=["metadatas", "documents"])
    assert float(chair["metadatas"][0]["price_usd"]) == pytest.approx(111)
    assert chair["documents"][0].endswith("Updated chair copy.")

    extra = dict(FIXTURE[2])
    extra["id"] = "lamp-extra"
    extra["category"] = "lighting"
    extra["name"] = "Extra Lamp"
    _write_catalog(tmp_path / "extra.json", [extra])
    assert ingest_catalog(tmp_path / "extra.json", chroma_dir, embedder=embedder) == 1
    assert get_collection(chroma_dir).count() == 4

    _write_catalog(tmp_path / "only-rug.json", [FIXTURE[2]])
    assert ingest_catalog(
        tmp_path / "only-rug.json",
        chroma_dir,
        rebuild=True,
        embedder=embedder,
    ) == 1
    collection = get_collection(chroma_dir)
    assert collection.count() == 1
    assert collection.get()["ids"] == ["rug-fjord"]


def test_rebuild_of_empty_catalog_clears_the_collection(tmp_path):
    catalog = _write_catalog(tmp_path / "catalog.json", FIXTURE)
    chroma_dir = tmp_path / "chroma"
    embedder = FakeEmbedder()
    ingest_catalog(catalog, chroma_dir, embedder=embedder)

    empty = _write_catalog(tmp_path / "empty.json", [])
    assert ingest_catalog(empty, chroma_dir, rebuild=True, embedder=embedder) == 0
    assert get_collection(chroma_dir).count() == 0


def test_ingest_uses_get_embedder_when_none_is_injected(tmp_path, monkeypatch):
    catalog = _write_catalog(tmp_path / "catalog.json", FIXTURE)
    chroma_dir = tmp_path / "chroma"
    monkeypatch.setattr("app.ingest.get_embedder", lambda: FakeEmbedder())

    assert ingest_catalog(catalog, chroma_dir, rebuild=True) == 3
    assert get_collection(chroma_dir).count() == 3


def test_default_paths_match_the_repo_layout():
    assert default_catalog_path() == CATALOG_PATH
    assert CATALOG_PATH.is_file()
    assert default_chroma_dir() == BACKEND_ROOT / ".chroma"


def test_cli_rebuild_uses_injected_embedder(tmp_path, monkeypatch, capsys):
    catalog = _write_catalog(tmp_path / "catalog.json", FIXTURE)
    chroma_dir = tmp_path / "chroma"
    monkeypatch.setattr("app.ingest.get_embedder", lambda: FakeEmbedder())

    exit_code = main([
        "--catalog",
        str(catalog),
        "--chroma-dir",
        str(chroma_dir),
        "--rebuild",
    ])

    assert exit_code == 0
    assert get_collection(chroma_dir).count() == 3
    assert "3" in capsys.readouterr().out


@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("RUN_EMBEDDING_INTEGRATION") != "1",
    reason="set RUN_EMBEDDING_INTEGRATION=1 to download google/embeddinggemma-300m",
)
def test_embeddinggemma_query_and_document_shapes():
    embedder = EmbeddingGemmaEmbedder()
    documents = embedder.embed_documents(["Oslo Sofa | sofas | modern | linen | a sofa"])
    query = embedder.embed_query("modern linen sofa")
    assert len(documents) == 1
    assert len(documents[0]) == 768
    assert len(query) == 768

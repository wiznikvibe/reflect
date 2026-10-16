"""Load the living-room catalog, embed it, and upsert it into Chroma."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import chromadb
from chromadb.config import Settings
from chromadb.errors import NotFoundError

from app.embeddings import Embedder, get_embedder
from app.schemas import Product

COLLECTION_NAME = "products"
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


def default_catalog_path() -> Path:
    """Repo catalog written by the schema task."""

    return Path(__file__).resolve().parents[2] / "data" / "catalog" / "living_room.json"


def default_chroma_dir() -> Path:
    """Persistent Chroma directory next to the backend package."""

    return Path(__file__).resolve().parents[1] / ".chroma"


def normalize_text(value: str) -> str:
    """Trim and collapse whitespace, including newlines."""

    return " ".join(value.split())


def _normalize_style_tags(tags: list[object]) -> list[object]:
    cleaned: list[object] = []
    seen: set[str] = set()
    for tag in tags:
        if not isinstance(tag, str):
            cleaned.append(tag)
            continue
        text = normalize_text(tag).lower()
        if not text or text in seen:
            continue
        seen.add(text)
        cleaned.append(text)
    return cleaned


def normalize_product(raw: dict) -> Product:
    """Clean one catalog object and validate it as a Product."""

    if not isinstance(raw, dict):
        raise ValueError("catalog item must be an object")
    data = dict(raw)
    for key in ("id", "name", "category", "material", "description"):
        value = data.get(key)
        if isinstance(value, str):
            data[key] = normalize_text(value)
    style_tags = data.get("style_tags")
    if isinstance(style_tags, list):
        data["style_tags"] = _normalize_style_tags(style_tags)
    return Product.model_validate(data)


def embed_text(product: Product) -> str:
    """Text sent to the embedder: name | category | styles | material | description."""

    styles = ", ".join(product.style_tags)
    return (
        f"{product.name} | {product.category} | {styles} | "
        f"{product.material} | {product.description}"
    )


def stringify_style_tags(tags: list[str]) -> str:
    """JSON list, because Chroma metadata values cannot be arrays."""

    return json.dumps(list(tags))


def parse_style_tags(value: str) -> list[str]:
    """Inverse of ``stringify_style_tags``."""

    parsed = json.loads(value)
    if not isinstance(parsed, list) or not all(isinstance(tag, str) for tag in parsed):
        raise ValueError("style_tags metadata must be a JSON list of strings")
    return parsed


def product_metadata(product: Product) -> dict[str, str | float]:
    """Flat metadata for Chroma filters. ``style_tags`` is a JSON string."""

    values: dict[str, str | float] = {
        "id": product.id,
        "category": product.category,
        "price_usd": float(product.price_usd),
        "width_cm": float(product.dimensions.width_cm),
        "depth_cm": float(product.dimensions.depth_cm),
        "height_cm": float(product.dimensions.height_cm),
        "style_tags": stringify_style_tags(product.style_tags),
        "material": product.material,
        "name": product.name,
    }
    return {key: values[key] for key in METADATA_KEYS}


def load_catalog(catalog_path: str | Path) -> list[Product]:
    """Read a catalog JSON list, normalize each item, and reject duplicate ids."""

    path = Path(catalog_path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"catalog must be a JSON list: {path}")
    products = [normalize_product(item) for item in raw]
    seen: set[str] = set()
    duplicates: list[str] = []
    for product in products:
        if product.id in seen and product.id not in duplicates:
            duplicates.append(product.id)
        seen.add(product.id)
    if duplicates:
        joined = ", ".join(duplicates)
        raise ValueError(f"duplicate product ids: {joined}")
    return products


def _client(chroma_dir: str | Path) -> chromadb.ClientAPI:
    path = Path(chroma_dir)
    path.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=str(path),
        settings=Settings(anonymized_telemetry=False),
    )


def get_collection(chroma_dir: str | Path):
    """Open the persisted product collection, creating it if needed.

    Pass ``embeddings`` / ``query_embeddings`` yourself. The collection does not
    run Chroma's default MiniLM model when vectors are supplied.
    """

    return _client(chroma_dir).get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def ingest_catalog(
    catalog_path: str | Path,
    chroma_dir: str | Path,
    rebuild: bool = False,
    *,
    embedder: Embedder | None = None,
) -> int:
    """Upsert the catalog and return how many products were written.

    ``rebuild=True`` deletes the collection first so ids removed from the
    catalog disappear. Without it, existing ids are updated and new ids are added.
    """

    products = load_catalog(catalog_path)
    texts = [embed_text(product) for product in products]
    if embedder is None:
        embedder = get_embedder()
    vectors = embedder.embed_documents(texts) if texts else []
    if len(vectors) != len(products):
        raise ValueError(
            f"embedder returned {len(vectors)} vectors for {len(products)} products"
        )

    client = _client(chroma_dir)
    if rebuild:
        try:
            client.delete_collection(COLLECTION_NAME)
        except NotFoundError:
            pass
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    if products:
        collection.upsert(
            ids=[product.id for product in products],
            embeddings=vectors,
            documents=texts,
            metadatas=[product_metadata(product) for product in products],
        )
    return len(products)


def main(argv: list[str] | None = None) -> int:
    """CLI entry: ``python -m app.ingest --rebuild`` from ``backend/``."""

    parser = argparse.ArgumentParser(
        description="Ingest the living-room catalog into Chroma."
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Delete the collection before upserting.",
    )
    parser.add_argument("--catalog", type=Path, default=None)
    parser.add_argument("--chroma-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    catalog = args.catalog if args.catalog is not None else default_catalog_path()
    chroma_dir = args.chroma_dir if args.chroma_dir is not None else default_chroma_dir()
    count = ingest_catalog(catalog, chroma_dir, rebuild=args.rebuild)
    print(f"upserted {count} products into {chroma_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

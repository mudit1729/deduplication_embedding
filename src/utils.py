"""Shared helpers for the duplicate detection toy project."""

from __future__ import annotations

import html
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import hnswlib
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
INDICES_DIR = PROJECT_ROOT / "indices"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_THRESHOLD = 0.80
DEFAULT_INDEX_FILE = INDICES_DIR / "quora_hnsw.index"
DEFAULT_INDEX_METADATA_FILE = INDICES_DIR / "quora_hnsw_metadata.json"
DEFAULT_MAPPING_FILE = INDICES_DIR / "id_mapping.csv"
DEFAULT_CORPUS_FILE = PROCESSED_DIR / "corpus.csv"
DEFAULT_TRAIN_FILE = PROCESSED_DIR / "train_pairs.csv"
DEFAULT_VALIDATION_FILE = PROCESSED_DIR / "validation_pairs.csv"
DEFAULT_EMBEDDINGS_FILE = MODELS_DIR / "corpus_embeddings.npy"
DEFAULT_EMBEDDING_METADATA_FILE = MODELS_DIR / "embedding_metadata.json"
DEFAULT_EVALUATION_SUMMARY_FILE = MODELS_DIR / "evaluation_summary.json"


@dataclass
class IndexArtifacts:
    """All assets required for ANN search."""

    index: hnswlib.Index
    corpus_df: pd.DataFrame
    corpus_embeddings: np.ndarray
    metadata: dict[str, Any]


def ensure_project_dirs() -> None:
    """Create the expected directory structure."""
    for directory in (
        DATA_DIR,
        RAW_DIR,
        PROCESSED_DIR,
        MODELS_DIR,
        INDICES_DIR,
        NOTEBOOKS_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)


def seed_everything(seed: int) -> None:
    """Seed common RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def minimal_normalize_text(text: str) -> str:
    """Apply lightweight text normalization suitable for a baseline."""
    normalized = html.unescape(str(text))
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def canonicalize_text(text: str) -> str:
    """Canonical text form used for comparisons and lookup keys."""
    return minimal_normalize_text(text).lower()


def pair_key(question1: str, question2: str) -> str:
    """Create an order-invariant pair key."""
    left, right = sorted((canonicalize_text(question1), canonicalize_text(question2)))
    return f"{left} || {right}"


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """L2-normalize embedding vectors."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.clip(norms, a_min=1e-12, a_max=None)
    return embeddings / norms


def detect_device(preferred_device: str | None = None) -> str:
    """Choose a local device, preferring MPS on Apple Silicon when available."""
    if preferred_device:
        return preferred_device
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_sentence_transformer(
    model_name_or_path: str | None = None,
    device: str | None = None,
) -> SentenceTransformer:
    """Load a sentence-transformer model."""
    resolved_model = model_name_or_path or DEFAULT_MODEL_NAME
    resolved_device = detect_device(device)
    return SentenceTransformer(resolved_model, device=resolved_device)


def encode_texts(
    model: SentenceTransformer,
    texts: Sequence[str],
    batch_size: int = 128,
    show_progress_bar: bool = True,
) -> np.ndarray:
    """Encode texts into normalized float32 embeddings."""
    if not texts:
        dimension = model.get_sentence_embedding_dimension()
        return np.empty((0, dimension), dtype=np.float32)

    embeddings = model.encode(
        list(texts),
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=show_progress_bar,
    )
    return embeddings.astype(np.float32, copy=False)


def save_json(payload: dict[str, Any], path: str | Path) -> None:
    """Write JSON with stable formatting."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def load_json(path: str | Path, default: dict[str, Any] | None = None) -> dict[str, Any]:
    """Load JSON if present, otherwise return a default value."""
    path = Path(path)
    if not path.exists():
        return default or {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_pairs(path: str | Path) -> pd.DataFrame:
    """Load a pair dataset from CSV."""
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing pair file: {csv_path}")
    return pd.read_csv(csv_path)


def load_corpus(path: str | Path = DEFAULT_CORPUS_FILE) -> pd.DataFrame:
    """Load the corpus CSV."""
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing corpus file: {csv_path}")
    return pd.read_csv(csv_path)


def load_index_artifacts(
    index_path: str | Path = DEFAULT_INDEX_FILE,
    metadata_path: str | Path = DEFAULT_INDEX_METADATA_FILE,
    corpus_path: str | Path = DEFAULT_CORPUS_FILE,
    embeddings_path: str | Path = DEFAULT_EMBEDDINGS_FILE,
    ef_search: int | None = None,
) -> IndexArtifacts:
    """Load the HNSW index and its aligned metadata."""
    index_path = Path(index_path)
    metadata_path = Path(metadata_path)
    corpus_path = Path(corpus_path)
    embeddings_path = Path(embeddings_path)

    metadata = load_json(metadata_path)
    corpus_df = load_corpus(corpus_path)
    corpus_embeddings = np.load(embeddings_path).astype(np.float32, copy=False)

    if len(corpus_df) != corpus_embeddings.shape[0]:
        raise ValueError(
            "Corpus rows and embedding rows do not match: "
            f"{len(corpus_df)} != {corpus_embeddings.shape[0]}"
        )

    space = metadata.get("space", "ip")
    dim = int(metadata.get("dim", corpus_embeddings.shape[1]))
    index = hnswlib.Index(space=space, dim=dim)
    index.load_index(str(index_path), max_elements=len(corpus_df))

    resolved_ef_search = ef_search or metadata.get("ef_search")
    if resolved_ef_search:
        index.set_ef(int(resolved_ef_search))

    return IndexArtifacts(
        index=index,
        corpus_df=corpus_df,
        corpus_embeddings=corpus_embeddings,
        metadata=metadata,
    )


def search_neighbors(
    artifacts: IndexArtifacts,
    query_embedding: np.ndarray,
    top_k: int = 5,
    exclude_question_ids: set[str] | None = None,
    exclude_text: str | None = None,
    buffer: int = 20,
) -> list[dict[str, Any]]:
    """Search the ANN index and return scored neighbors."""
    if len(artifacts.corpus_df) == 0:
        return []

    query_vector = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
    excluded_ids = set(exclude_question_ids or set())
    excluded_text = canonicalize_text(exclude_text) if exclude_text else None

    requested = top_k + max(buffer, 0) + len(excluded_ids)
    search_width = min(len(artifacts.corpus_df), max(top_k, requested))
    ann_ids, _distances = artifacts.index.knn_query(query_vector, k=search_width)

    results: list[dict[str, Any]] = []
    for ann_id in ann_ids[0].tolist():
        row = artifacts.corpus_df.iloc[int(ann_id)]
        question_id = str(row["question_id"])
        question_text = str(row["question_text"])

        if question_id in excluded_ids:
            continue
        if excluded_text and canonicalize_text(question_text) == excluded_text:
            continue

        similarity = float(np.dot(query_vector[0], artifacts.corpus_embeddings[int(ann_id)]))
        results.append(
            {
                "ann_id": int(ann_id),
                "question_id": question_id,
                "question_text": question_text,
                "score": float(np.clip(similarity, -1.0, 1.0)),
            }
        )

    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:top_k]


def load_best_threshold(
    summary_path: str | Path = DEFAULT_EVALUATION_SUMMARY_FILE,
    fallback: float = DEFAULT_THRESHOLD,
) -> float:
    """Load the best threshold from evaluation output if available."""
    summary = load_json(summary_path)
    best_threshold = summary.get("best_threshold")
    if best_threshold is None:
        return fallback
    return float(best_threshold)

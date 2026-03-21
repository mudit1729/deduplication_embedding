"""Build an HNSW ANN index over the corpus embeddings."""

from __future__ import annotations

import argparse
from pathlib import Path

import hnswlib
import numpy as np

from utils import (
    DEFAULT_CORPUS_FILE,
    DEFAULT_EMBEDDING_METADATA_FILE,
    DEFAULT_EMBEDDINGS_FILE,
    DEFAULT_INDEX_FILE,
    DEFAULT_INDEX_METADATA_FILE,
    DEFAULT_MAPPING_FILE,
    DEFAULT_THRESHOLD,
    load_corpus,
    load_json,
    normalize_embeddings,
    save_json,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus_file", type=Path, default=DEFAULT_CORPUS_FILE)
    parser.add_argument("--embeddings_file", type=Path, default=DEFAULT_EMBEDDINGS_FILE)
    parser.add_argument(
        "--embedding_metadata_file",
        type=Path,
        default=DEFAULT_EMBEDDING_METADATA_FILE,
    )
    parser.add_argument("--index_file", type=Path, default=DEFAULT_INDEX_FILE)
    parser.add_argument("--metadata_file", type=Path, default=DEFAULT_INDEX_METADATA_FILE)
    parser.add_argument("--mapping_file", type=Path, default=DEFAULT_MAPPING_FILE)
    parser.add_argument("--space", choices=["ip", "cosine"], default="ip")
    parser.add_argument("--m", type=int, default=32)
    parser.add_argument("--ef_construction", type=int, default=200)
    parser.add_argument("--ef_search", type=int, default=64)
    parser.add_argument("--default_threshold", type=float, default=DEFAULT_THRESHOLD)
    return parser.parse_args()


def main() -> None:
    """Build and persist the ANN index."""
    args = parse_args()

    corpus_df = load_corpus(args.corpus_file)
    embeddings = np.load(args.embeddings_file).astype(np.float32, copy=False)
    embeddings = normalize_embeddings(embeddings)

    if len(corpus_df) != embeddings.shape[0]:
        raise ValueError(
            "Corpus rows and embedding rows do not match: "
            f"{len(corpus_df)} != {embeddings.shape[0]}"
        )

    index = hnswlib.Index(space=args.space, dim=embeddings.shape[1])
    index.init_index(
        max_elements=len(corpus_df),
        ef_construction=args.ef_construction,
        M=args.m,
    )
    index.add_items(embeddings, np.arange(len(corpus_df)))
    index.set_ef(args.ef_search)

    args.index_file.parent.mkdir(parents=True, exist_ok=True)
    index.save_index(str(args.index_file))

    mapping_df = corpus_df.copy()
    mapping_df.insert(0, "ann_id", range(len(mapping_df)))
    mapping_df.to_csv(args.mapping_file, index=False)

    embedding_metadata = load_json(args.embedding_metadata_file)
    metadata = {
        "corpus_file": str(args.corpus_file),
        "default_threshold": float(args.default_threshold),
        "dim": int(embeddings.shape[1]),
        "ef_construction": int(args.ef_construction),
        "ef_search": int(args.ef_search),
        "embeddings_file": str(args.embeddings_file),
        "embedding_model_name_or_path": embedding_metadata.get("model_name_or_path"),
        "index_file": str(args.index_file),
        "m": int(args.m),
        "mapping_file": str(args.mapping_file),
        "num_elements": int(len(corpus_df)),
        "space": args.space,
    }
    save_json(metadata, args.metadata_file)

    print(f"Saved index to: {args.index_file}")
    print(f"Saved index metadata to: {args.metadata_file}")
    print(f"Saved ANN ID mapping to: {args.mapping_file}")
    print(metadata)


if __name__ == "__main__":
    main()

"""Run local ANN search for near-duplicate questions."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from utils import (
    DEFAULT_CORPUS_FILE,
    DEFAULT_EMBEDDINGS_FILE,
    DEFAULT_EVALUATION_SUMMARY_FILE,
    DEFAULT_INDEX_FILE,
    DEFAULT_INDEX_METADATA_FILE,
    DEFAULT_MODEL_NAME,
    encode_texts,
    load_best_threshold,
    load_index_artifacts,
    load_sentence_transformer,
    search_neighbors,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", required=True, help="Free-text query question.")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--threshold_summary_file", type=Path, default=DEFAULT_EVALUATION_SUMMARY_FILE)
    parser.add_argument("--model_name_or_path", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--device", default=None)
    parser.add_argument("--index_file", type=Path, default=DEFAULT_INDEX_FILE)
    parser.add_argument("--index_metadata_file", type=Path, default=DEFAULT_INDEX_METADATA_FILE)
    parser.add_argument("--corpus_file", type=Path, default=DEFAULT_CORPUS_FILE)
    parser.add_argument("--embeddings_file", type=Path, default=DEFAULT_EMBEDDINGS_FILE)
    parser.add_argument("--ef_search", type=int, default=None)
    parser.add_argument("--search_buffer", type=int, default=20)
    parser.add_argument(
        "--allow_exact_text_match",
        action="store_true",
        help="Return exact text matches instead of filtering them out.",
    )
    return parser.parse_args()


def main() -> None:
    """Search the local ANN index and print a readable result table."""
    args = parse_args()

    threshold = (
        args.threshold
        if args.threshold is not None
        else load_best_threshold(args.threshold_summary_file)
    )

    artifacts = load_index_artifacts(
        index_path=args.index_file,
        metadata_path=args.index_metadata_file,
        corpus_path=args.corpus_file,
        embeddings_path=args.embeddings_file,
        ef_search=args.ef_search,
    )
    model = load_sentence_transformer(args.model_name_or_path, args.device)
    query_embedding = encode_texts(
        model,
        [args.query],
        batch_size=1,
        show_progress_bar=False,
    )[0]

    results = search_neighbors(
        artifacts=artifacts,
        query_embedding=query_embedding,
        top_k=args.top_k,
        exclude_text=None if args.allow_exact_text_match else args.query,
        buffer=args.search_buffer,
    )

    if not results:
        print("No matches found.")
        return

    top_score = float(results[0]["score"])
    decision = "duplicate" if top_score >= threshold else "non-duplicate"

    display_df = pd.DataFrame(results)
    display_df.insert(0, "rank", range(1, len(display_df) + 1))
    display_df["prediction"] = display_df["score"].apply(
        lambda score: "duplicate" if score >= threshold else "non-duplicate"
    )
    display_df["score"] = display_df["score"].map(lambda value: round(float(value), 4))

    print(f"Query: {args.query}")
    print(f"Decision threshold: {threshold:.4f}")
    print(f"Top match score: {top_score:.4f}")
    print(f"Pipeline decision: {decision}")
    print()
    print(display_df.to_string(index=False))


if __name__ == "__main__":
    main()

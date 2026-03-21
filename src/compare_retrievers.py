"""Compare multiple retrieval mechanisms on the held-out validation set."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm.auto import tqdm

from utils import (
    DEFAULT_CORPUS_FILE,
    DEFAULT_EMBEDDINGS_FILE,
    DEFAULT_INDEX_FILE,
    DEFAULT_INDEX_METADATA_FILE,
    DEFAULT_VALIDATION_FILE,
    INDICES_DIR,
    MODELS_DIR,
    load_corpus,
    load_index_artifacts,
    load_pairs,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation_file", type=Path, default=DEFAULT_VALIDATION_FILE)
    parser.add_argument("--corpus_file", type=Path, default=DEFAULT_CORPUS_FILE)
    parser.add_argument("--embeddings_file", type=Path, default=DEFAULT_EMBEDDINGS_FILE)
    parser.add_argument("--hnsw_index_file", type=Path, default=DEFAULT_INDEX_FILE)
    parser.add_argument("--hnsw_index_metadata_file", type=Path, default=DEFAULT_INDEX_METADATA_FILE)
    parser.add_argument("--faiss_index_file", type=Path, default=INDICES_DIR / "quora_faiss.index")
    parser.add_argument(
        "--faiss_index_metadata_file",
        type=Path,
        default=INDICES_DIR / "quora_faiss_metadata.json",
    )
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--search_buffer", type=int, default=20)
    parser.add_argument(
        "--hybrid_candidate_k",
        type=int,
        default=20,
        help="How many candidates to collect from each retriever before fusion.",
    )
    parser.add_argument(
        "--hybrid_dense_weight",
        type=float,
        default=0.7,
        help="Dense score weight in the hybrid fusion formula.",
    )
    parser.add_argument(
        "--summary_csv",
        type=Path,
        default=MODELS_DIR / "retriever_comparison.csv",
    )
    parser.add_argument(
        "--details_csv",
        type=Path,
        default=MODELS_DIR / "retriever_comparison_details.csv",
    )
    return parser.parse_args()


def top_indices_from_dense_scores(
    scores: np.ndarray,
    exclude_idx: int | None,
    top_n: int,
) -> np.ndarray:
    """Return top indices from a dense score vector."""
    working_scores = scores.copy()
    if exclude_idx is not None and 0 <= exclude_idx < len(working_scores):
        working_scores[exclude_idx] = -np.inf

    top_n = min(top_n, len(working_scores))
    if top_n <= 0:
        return np.empty(0, dtype=np.int64)

    candidate_indices = np.argpartition(working_scores, -top_n)[-top_n:]
    ordered = candidate_indices[np.argsort(working_scores[candidate_indices])[::-1]]
    return ordered.astype(np.int64, copy=False)


def minmax_normalize(scores: dict[int, float]) -> dict[int, float]:
    """Scale scores to [0, 1] within a candidate set."""
    if not scores:
        return {}
    values = np.asarray(list(scores.values()), dtype=np.float32)
    lower = float(values.min())
    upper = float(values.max())
    if upper - lower < 1e-12:
        return {key: 1.0 for key in scores}
    return {key: (float(value) - lower) / (upper - lower) for key, value in scores.items()}


def compute_retrieval_metrics(rank_series: pd.Series, top_k: int) -> dict[str, float]:
    """Compute Recall@1, Recall@k, and MRR from target ranks."""
    hits_at_1 = rank_series.eq(1).astype(int)
    hits_at_k = rank_series.apply(lambda rank: int(pd.notna(rank) and int(rank) <= top_k))
    reciprocal_ranks = rank_series.apply(
        lambda rank: 0.0 if pd.isna(rank) else 1.0 / int(rank)
    )
    return {
        "evaluated_queries": int(len(rank_series)),
        "recall_at_1": float(hits_at_1.mean()),
        f"recall_at_{top_k}": float(hits_at_k.mean()),
        "mrr": float(reciprocal_ranks.mean()),
    }


def main() -> None:
    """Run the retrieval comparison."""
    args = parse_args()

    corpus_df = load_corpus(args.corpus_file)
    validation_df = load_pairs(args.validation_file)
    corpus_embeddings = np.load(args.embeddings_file).astype(np.float32, copy=False)

    if len(corpus_df) != len(corpus_embeddings):
        raise ValueError("Corpus rows and embedding rows do not match.")

    question_id_to_idx = {
        str(question_id): idx for idx, question_id in enumerate(corpus_df["question_id"].tolist())
    }
    positive_df = validation_df[validation_df["label"] == 1].copy()
    positive_df = positive_df[
        positive_df["question1_id"].notna() & positive_df["question2_id"].notna()
    ].reset_index(drop=True)

    hnsw_artifacts = load_index_artifacts(
        index_path=args.hnsw_index_file,
        metadata_path=args.hnsw_index_metadata_file,
        corpus_path=args.corpus_file,
        embeddings_path=args.embeddings_file,
    )

    faiss_artifacts = None
    if args.faiss_index_file.exists() and args.faiss_index_metadata_file.exists():
        faiss_artifacts = load_index_artifacts(
            index_path=args.faiss_index_file,
            metadata_path=args.faiss_index_metadata_file,
            corpus_path=args.corpus_file,
            embeddings_path=args.embeddings_file,
        )

    corpus_texts = corpus_df["question_text"].astype(str).tolist()
    word_vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
    )
    char_vectorizer = TfidfVectorizer(
        lowercase=True,
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        sublinear_tf=True,
    )
    word_matrix = word_vectorizer.fit_transform(corpus_texts)
    char_matrix = char_vectorizer.fit_transform(corpus_texts)

    records: list[dict[str, object]] = []
    for row in tqdm(positive_df.itertuples(index=False), total=len(positive_df), desc="Retrievers"):
        query_idx = question_id_to_idx[str(row.question1_id)]
        target_idx = question_id_to_idx[str(row.question2_id)]
        query_vector = corpus_embeddings[query_idx].reshape(1, -1)

        search_width = min(len(corpus_df), max(args.top_k, args.top_k + args.search_buffer + 1))
        hnsw_ids, _ = hnsw_artifacts.index.knn_query(query_vector, k=search_width)
        hnsw_scores = {
            int(idx): float(np.dot(query_vector[0], corpus_embeddings[int(idx)]))
            for idx in hnsw_ids[0].tolist()
            if int(idx) != query_idx
        }
        ordered_hnsw_ids = [
            idx
            for idx, _score in sorted(hnsw_scores.items(), key=lambda item: item[1], reverse=True)
        ][: args.top_k]

        dense_scores = corpus_embeddings @ query_vector[0]
        ordered_exact_ids = top_indices_from_dense_scores(
            dense_scores,
            exclude_idx=query_idx,
            top_n=args.top_k,
        ).tolist()

        if faiss_artifacts is not None:
            faiss_scores, faiss_ids = faiss_artifacts.index.search(query_vector, search_width)
            faiss_candidates = [
                int(idx) for idx in faiss_ids[0].tolist() if int(idx) >= 0 and int(idx) != query_idx
            ]
            ordered_faiss_ids = faiss_candidates[: args.top_k]
        else:
            ordered_faiss_ids = ordered_exact_ids

        word_scores = (word_matrix[query_idx] @ word_matrix.T).toarray().ravel()
        ordered_word_ids = top_indices_from_dense_scores(
            word_scores,
            exclude_idx=query_idx,
            top_n=args.top_k,
        ).tolist()

        char_scores = (char_matrix[query_idx] @ char_matrix.T).toarray().ravel()
        ordered_char_ids = top_indices_from_dense_scores(
            char_scores,
            exclude_idx=query_idx,
            top_n=args.top_k,
        ).tolist()

        dense_candidate_ids = top_indices_from_dense_scores(
            dense_scores,
            exclude_idx=query_idx,
            top_n=args.hybrid_candidate_k,
        ).tolist()
        char_candidate_ids = top_indices_from_dense_scores(
            char_scores,
            exclude_idx=query_idx,
            top_n=args.hybrid_candidate_k,
        ).tolist()
        hybrid_candidate_ids = list(dict.fromkeys(dense_candidate_ids + char_candidate_ids))

        dense_component = minmax_normalize({idx: float(dense_scores[idx]) for idx in hybrid_candidate_ids})
        char_component = minmax_normalize({idx: float(char_scores[idx]) for idx in hybrid_candidate_ids})
        hybrid_scores = {
            idx: (
                args.hybrid_dense_weight * dense_component.get(idx, 0.0)
                + (1.0 - args.hybrid_dense_weight) * char_component.get(idx, 0.0)
            )
            for idx in hybrid_candidate_ids
        }
        ordered_hybrid_ids = [
            idx
            for idx, _score in sorted(hybrid_scores.items(), key=lambda item: item[1], reverse=True)
        ][: args.top_k]

        def find_rank(candidate_ids: list[int]) -> int | None:
            return next(
                (rank for rank, candidate_idx in enumerate(candidate_ids, start=1) if candidate_idx == target_idx),
                None,
            )

        records.append(
            {
                "pair_id": row.pair_id,
                "question1": row.question1,
                "question2": row.question2,
                "dense_hnsw_rank": find_rank(ordered_hnsw_ids),
                "dense_exact_rank": find_rank(ordered_exact_ids),
                "dense_faiss_rank": find_rank(ordered_faiss_ids),
                "tfidf_word_rank": find_rank(ordered_word_ids),
                "tfidf_char_rank": find_rank(ordered_char_ids),
                "hybrid_dense_char_rank": find_rank(ordered_hybrid_ids),
            }
        )

    details_df = pd.DataFrame.from_records(records)

    method_columns = {
        "dense_hnsw": "dense_hnsw_rank",
        "dense_exact": "dense_exact_rank",
        "dense_faiss": "dense_faiss_rank",
        "tfidf_word": "tfidf_word_rank",
        "tfidf_char": "tfidf_char_rank",
        "hybrid_dense_char": "hybrid_dense_char_rank",
    }

    summary_records = []
    for method_name, column_name in method_columns.items():
        if column_name not in details_df.columns:
            continue
        metrics = compute_retrieval_metrics(details_df[column_name], args.top_k)
        metrics["method"] = method_name
        summary_records.append(metrics)

    summary_df = (
        pd.DataFrame.from_records(summary_records)
        .sort_values(["recall_at_1", f"recall_at_{args.top_k}", "mrr"], ascending=False)
        .reset_index(drop=True)
    )

    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(args.summary_csv, index=False)
    details_df.to_csv(args.details_csv, index=False)

    print(f"Saved retriever summary to: {args.summary_csv}")
    print(f"Saved per-query details to: {args.details_csv}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()

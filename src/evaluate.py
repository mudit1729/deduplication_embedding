"""Evaluate retrieval and threshold-based duplicate classification."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_fscore_support
from tqdm.auto import tqdm

from utils import (
    DEFAULT_CORPUS_FILE,
    DEFAULT_EMBEDDINGS_FILE,
    DEFAULT_EVALUATION_SUMMARY_FILE,
    DEFAULT_INDEX_FILE,
    DEFAULT_INDEX_METADATA_FILE,
    DEFAULT_MODEL_NAME,
    DEFAULT_VALIDATION_FILE,
    MODELS_DIR,
    encode_texts,
    load_index_artifacts,
    load_pairs,
    load_sentence_transformer,
    save_json,
    search_neighbors,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation_file", type=Path, default=DEFAULT_VALIDATION_FILE)
    parser.add_argument("--corpus_file", type=Path, default=DEFAULT_CORPUS_FILE)
    parser.add_argument("--index_file", type=Path, default=DEFAULT_INDEX_FILE)
    parser.add_argument("--index_metadata_file", type=Path, default=DEFAULT_INDEX_METADATA_FILE)
    parser.add_argument("--embeddings_file", type=Path, default=DEFAULT_EMBEDDINGS_FILE)
    parser.add_argument("--model_name_or_path", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--search_buffer", type=int, default=20)
    parser.add_argument("--min_threshold", type=float, default=0.30)
    parser.add_argument("--max_threshold", type=float, default=0.95)
    parser.add_argument("--threshold_step", type=float, default=0.01)
    parser.add_argument(
        "--threshold_csv",
        type=Path,
        default=MODELS_DIR / "validation_threshold_sweep.csv",
    )
    parser.add_argument(
        "--pair_scores_csv",
        type=Path,
        default=MODELS_DIR / "validation_pair_scores.csv",
    )
    parser.add_argument(
        "--retrieval_csv",
        type=Path,
        default=MODELS_DIR / "validation_retrieval_results.csv",
    )
    parser.add_argument(
        "--summary_json",
        type=Path,
        default=DEFAULT_EVALUATION_SUMMARY_FILE,
    )
    return parser.parse_args()


def build_embedding_lookup(texts: pd.Series, model_name_or_path: str, device: str | None, batch_size: int):
    """Encode unique texts once and map them back by string."""
    unique_texts = pd.Index(texts).drop_duplicates().tolist()
    model = load_sentence_transformer(model_name_or_path, device)
    embeddings = encode_texts(
        model,
        unique_texts,
        batch_size=batch_size,
        show_progress_bar=True,
    )
    lookup = {text: embedding for text, embedding in zip(unique_texts, embeddings)}
    return model, lookup


def compute_pair_scores(validation_df: pd.DataFrame, embedding_lookup: dict[str, np.ndarray]) -> np.ndarray:
    """Compute cosine similarities for the held-out pairs."""
    scores = np.empty(len(validation_df), dtype=np.float32)
    for index, row in enumerate(validation_df.itertuples(index=False)):
        scores[index] = float(np.dot(embedding_lookup[row.question1], embedding_lookup[row.question2]))
    return scores


def compute_binary_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    """Compute classification metrics at a given threshold."""
    predictions = (scores >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="binary",
        zero_division=0,
    )
    true_positive = int(((predictions == 1) & (labels == 1)).sum())
    false_positive = int(((predictions == 1) & (labels == 0)).sum())
    false_negative = int(((predictions == 0) & (labels == 1)).sum())
    true_negative = int(((predictions == 0) & (labels == 0)).sum())
    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
    }


def sweep_thresholds(
    labels: np.ndarray,
    scores: np.ndarray,
    min_threshold: float,
    max_threshold: float,
    step: float,
) -> pd.DataFrame:
    """Run a threshold sweep over cosine similarity values."""
    thresholds = np.arange(min_threshold, max_threshold + step / 2, step)
    records = [compute_binary_metrics(labels, scores, float(threshold)) for threshold in thresholds]
    return pd.DataFrame.from_records(records)


def evaluate_retrieval(
    validation_df: pd.DataFrame,
    embedding_lookup: dict[str, np.ndarray],
    top_k: int,
    search_buffer: int,
    artifacts,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Evaluate Recall@1, Recall@k, and MRR over positive validation pairs."""
    positive_df = validation_df[validation_df["label"] == 1].copy()
    positive_df = positive_df[
        positive_df["question1_id"].notna() & positive_df["question2_id"].notna()
    ].reset_index(drop=True)

    retrieval_records = []
    for row in tqdm(positive_df.itertuples(index=False), total=len(positive_df), desc="Retrieval"):
        exclude_ids = set()
        exclude_text = None
        if row.question1_id != row.question2_id:
            exclude_ids.add(str(row.question1_id))
            exclude_text = row.question1

        results = search_neighbors(
            artifacts=artifacts,
            query_embedding=embedding_lookup[row.question1],
            top_k=top_k,
            exclude_question_ids=exclude_ids,
            exclude_text=exclude_text,
            buffer=search_buffer,
        )
        rank = next(
            (
                result_rank
                for result_rank, result in enumerate(results, start=1)
                if result["question_id"] == row.question2_id
            ),
            None,
        )
        retrieval_records.append(
            {
                "pair_id": row.pair_id,
                "question1": row.question1,
                "question2": row.question2,
                "question1_id": row.question1_id,
                "question2_id": row.question2_id,
                "target_rank": rank,
                "hit_at_1": int(rank == 1) if rank is not None else 0,
                f"hit_at_{top_k}": int(rank is not None and rank <= top_k),
                "reciprocal_rank": 0.0 if rank is None else 1.0 / rank,
            }
        )

    retrieval_df = pd.DataFrame.from_records(retrieval_records)
    if retrieval_df.empty:
        metrics = {"evaluated_queries": 0, "recall_at_1": 0.0, f"recall_at_{top_k}": 0.0, "mrr": 0.0}
    else:
        metrics = {
            "evaluated_queries": int(len(retrieval_df)),
            "recall_at_1": float(retrieval_df["hit_at_1"].mean()),
            f"recall_at_{top_k}": float(retrieval_df[f"hit_at_{top_k}"].mean()),
            "mrr": float(retrieval_df["reciprocal_rank"].mean()),
        }
    return metrics, retrieval_df


def main() -> None:
    """Run evaluation for retrieval and pair classification."""
    args = parse_args()

    validation_df = load_pairs(args.validation_file)
    texts = pd.concat([validation_df["question1"], validation_df["question2"]], ignore_index=True)
    model, embedding_lookup = build_embedding_lookup(
        texts=texts,
        model_name_or_path=args.model_name_or_path,
        device=args.device,
        batch_size=args.batch_size,
    )
    _ = model

    pair_scores = compute_pair_scores(validation_df, embedding_lookup)
    labels = validation_df["label"].astype(int).to_numpy()
    threshold_df = sweep_thresholds(
        labels=labels,
        scores=pair_scores,
        min_threshold=args.min_threshold,
        max_threshold=args.max_threshold,
        step=args.threshold_step,
    ).sort_values(["f1", "recall", "precision"], ascending=False)

    best_metrics = threshold_df.iloc[0].to_dict()
    best_threshold = float(best_metrics["threshold"])
    threshold_df = threshold_df.sort_values("threshold").reset_index(drop=True)

    validation_scores_df = validation_df.copy()
    validation_scores_df["cosine_similarity"] = pair_scores
    validation_scores_df["predicted_label"] = (pair_scores >= best_threshold).astype(int)

    artifacts = load_index_artifacts(
        index_path=args.index_file,
        metadata_path=args.index_metadata_file,
        corpus_path=args.corpus_file,
        embeddings_path=args.embeddings_file,
    )
    retrieval_metrics, retrieval_df = evaluate_retrieval(
        validation_df=validation_df,
        embedding_lookup=embedding_lookup,
        top_k=args.top_k,
        search_buffer=args.search_buffer,
        artifacts=artifacts,
    )

    average_precision = float(average_precision_score(labels, pair_scores))
    summary = {
        "model_name_or_path": args.model_name_or_path,
        "average_precision": average_precision,
        "best_threshold": best_threshold,
        "classification_at_best_threshold": {
            key: float(value) if isinstance(value, (int, float, np.floating)) else value
            for key, value in best_metrics.items()
        },
        "retrieval": retrieval_metrics,
        "files": {
            "pair_scores_csv": str(args.pair_scores_csv),
            "retrieval_csv": str(args.retrieval_csv),
            "summary_json": str(args.summary_json),
            "threshold_csv": str(args.threshold_csv),
        },
    }

    args.threshold_csv.parent.mkdir(parents=True, exist_ok=True)
    threshold_df.to_csv(args.threshold_csv, index=False)
    validation_scores_df.to_csv(args.pair_scores_csv, index=False)
    retrieval_df.to_csv(args.retrieval_csv, index=False)
    save_json(summary, args.summary_json)

    print(f"Saved threshold sweep to: {args.threshold_csv}")
    print(f"Saved pair scores to: {args.pair_scores_csv}")
    print(f"Saved retrieval results to: {args.retrieval_csv}")
    print(f"Saved evaluation summary to: {args.summary_json}")
    print(summary)


if __name__ == "__main__":
    main()

"""Simulate a simple self-improvement loop from model errors."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from utils import (
    DEFAULT_CORPUS_FILE,
    DEFAULT_EMBEDDINGS_FILE,
    DEFAULT_EVALUATION_SUMMARY_FILE,
    DEFAULT_INDEX_FILE,
    DEFAULT_INDEX_METADATA_FILE,
    DEFAULT_MODEL_NAME,
    DEFAULT_TRAIN_FILE,
    DEFAULT_VALIDATION_FILE,
    PROCESSED_DIR,
    encode_texts,
    load_best_threshold,
    load_index_artifacts,
    load_pairs,
    load_sentence_transformer,
    pair_key,
    search_neighbors,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_file", type=Path, default=DEFAULT_TRAIN_FILE)
    parser.add_argument("--validation_file", type=Path, default=DEFAULT_VALIDATION_FILE)
    parser.add_argument("--corpus_file", type=Path, default=DEFAULT_CORPUS_FILE)
    parser.add_argument("--index_file", type=Path, default=DEFAULT_INDEX_FILE)
    parser.add_argument("--index_metadata_file", type=Path, default=DEFAULT_INDEX_METADATA_FILE)
    parser.add_argument("--embeddings_file", type=Path, default=DEFAULT_EMBEDDINGS_FILE)
    parser.add_argument("--model_name_or_path", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument(
        "--threshold_summary_file",
        type=Path,
        default=DEFAULT_EVALUATION_SUMMARY_FILE,
    )
    parser.add_argument(
        "--hard_negative_anchor_file",
        type=Path,
        default=None,
        help="Optional pair file whose questions are used as ANN mining anchors.",
    )
    parser.add_argument(
        "--hard_negative_positive_only",
        action="store_true",
        help="Restrict hard-negative anchors to positive duplicate pairs from the anchor file.",
    )
    parser.add_argument(
        "--hard_negative_use_both_question_columns",
        action="store_true",
        help="Mine from both question columns instead of only question1 anchors.",
    )
    parser.add_argument("--search_k", type=int, default=15)
    parser.add_argument("--search_buffer", type=int, default=30)
    parser.add_argument("--hard_negative_min_score", type=float, default=0.60)
    parser.add_argument(
        "--false_positives_output",
        type=Path,
        default=PROCESSED_DIR / "false_positives.csv",
    )
    parser.add_argument(
        "--false_negatives_output",
        type=Path,
        default=PROCESSED_DIR / "false_negatives.csv",
    )
    parser.add_argument(
        "--hard_negatives_output",
        type=Path,
        default=PROCESSED_DIR / "hard_negatives.csv",
    )
    parser.add_argument(
        "--feedback_output",
        type=Path,
        default=PROCESSED_DIR / "active_learning_feedback_examples.csv",
    )
    parser.add_argument(
        "--updated_train_output",
        type=Path,
        default=PROCESSED_DIR / "train_pairs_active_learning.csv",
    )
    return parser.parse_args()


def build_embedding_lookup(texts: pd.Series, model_name_or_path: str, device: str | None, batch_size: int):
    """Encode unique texts once."""
    unique_texts = pd.Index(texts).drop_duplicates().tolist()
    model = load_sentence_transformer(model_name_or_path, device)
    embeddings = encode_texts(
        model,
        unique_texts,
        batch_size=batch_size,
        show_progress_bar=True,
    )
    return {text: embedding for text, embedding in zip(unique_texts, embeddings)}


def compute_pair_scores(validation_df: pd.DataFrame, embedding_lookup: dict[str, np.ndarray]) -> np.ndarray:
    """Compute pair similarities for feedback mining."""
    scores = np.empty(len(validation_df), dtype=np.float32)
    for index, row in enumerate(validation_df.itertuples(index=False)):
        scores[index] = float(np.dot(embedding_lookup[row.question1], embedding_lookup[row.question2]))
    return scores


def build_hard_negative_queries(
    anchor_df: pd.DataFrame,
    positive_only: bool,
    use_both_question_columns: bool,
) -> pd.DataFrame:
    """Build the set of query anchors used for hard-negative mining."""
    working_df = anchor_df.copy()
    if positive_only:
        if "label" not in working_df.columns:
            raise ValueError("The hard-negative anchor file must contain a label column when using --hard_negative_positive_only.")
        working_df = working_df[working_df["label"] == 1].copy()

    query_frames = [
        working_df[["question1", "question1_id"]].rename(
            columns={"question1": "question_text", "question1_id": "question_id"}
        )
    ]
    if use_both_question_columns:
        query_frames.append(
            working_df[["question2", "question2_id"]].rename(
                columns={"question2": "question_text", "question2_id": "question_id"}
            )
        )

    query_df = pd.concat(query_frames, ignore_index=True)
    query_df = query_df.dropna(subset=["question_text"]).copy()
    query_df["question_text"] = query_df["question_text"].astype(str)
    query_df = query_df[query_df["question_text"] != ""].drop_duplicates().reset_index(drop=True)
    return query_df


def mine_hard_negatives(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    query_df: pd.DataFrame,
    embedding_lookup: dict[str, np.ndarray],
    artifacts,
    search_k: int,
    search_buffer: int,
    min_score: float,
) -> pd.DataFrame:
    """Mine close ANN neighbors that are labeled non-duplicate elsewhere."""
    negative_df = pd.concat([train_df, validation_df], ignore_index=True)
    negative_df = negative_df[negative_df["label"] == 0].copy()
    negative_lookup = {
        pair_key(row.question1, row.question2): row
        for row in negative_df.itertuples(index=False)
    }

    records = []
    for row in tqdm(query_df.itertuples(index=False), total=len(query_df), desc="Hard negatives"):
        query_text = str(row.question_text)
        query_id = getattr(row, "question_id", None)
        exclude_ids = {str(query_id)} if pd.notna(query_id) else set()
        results = search_neighbors(
            artifacts=artifacts,
            query_embedding=embedding_lookup[query_text],
            top_k=search_k,
            exclude_question_ids=exclude_ids,
            exclude_text=query_text,
            buffer=search_buffer,
        )
        for rank, result in enumerate(results, start=1):
            key = pair_key(query_text, result["question_text"])
            negative_example = negative_lookup.get(key)
            if negative_example is None or result["score"] < min_score:
                continue
            records.append(
                {
                    "question1": query_text,
                    "question1_id": query_id,
                    "question2": result["question_text"],
                    "question2_id": result["question_id"],
                    "similarity": float(result["score"]),
                    "ann_rank": rank,
                    "label": 0,
                    "feedback_type": "hard_negative",
                    "source_pair_id": negative_example.pair_id,
                }
            )

    if not records:
        return pd.DataFrame(
            columns=[
                "question1",
                "question1_id",
                "question2",
                "question2_id",
                "similarity",
                "ann_rank",
                "label",
                "feedback_type",
                "source_pair_id",
            ]
        )

    hard_negative_df = (
        pd.DataFrame.from_records(records)
        .sort_values(["similarity", "ann_rank"], ascending=[False, True])
        .drop_duplicates(subset=["question1", "question2"])
        .reset_index(drop=True)
    )
    return hard_negative_df


def main() -> None:
    """Run the active-learning style export."""
    args = parse_args()
    threshold = (
        args.threshold
        if args.threshold is not None
        else load_best_threshold(args.threshold_summary_file)
    )

    train_df = load_pairs(args.train_file)
    validation_df = load_pairs(args.validation_file)
    hard_negative_anchor_file = args.hard_negative_anchor_file or args.validation_file
    hard_negative_anchor_df = load_pairs(hard_negative_anchor_file)
    hard_negative_query_df = build_hard_negative_queries(
        anchor_df=hard_negative_anchor_df,
        positive_only=args.hard_negative_positive_only,
        use_both_question_columns=args.hard_negative_use_both_question_columns,
    )

    texts = pd.concat(
        [
            validation_df["question1"],
            validation_df["question2"],
            hard_negative_query_df["question_text"],
        ],
        ignore_index=True,
    )
    embedding_lookup = build_embedding_lookup(
        texts=texts,
        model_name_or_path=args.model_name_or_path,
        device=args.device,
        batch_size=args.batch_size,
    )

    pair_scores = compute_pair_scores(validation_df, embedding_lookup)
    predicted_labels = (pair_scores >= threshold).astype(int)
    scored_validation_df = validation_df.copy()
    scored_validation_df["similarity"] = pair_scores
    scored_validation_df["predicted_label"] = predicted_labels

    false_positives_df = scored_validation_df[
        (scored_validation_df["label"] == 0) & (scored_validation_df["predicted_label"] == 1)
    ].copy()
    false_positives_df["feedback_type"] = "false_positive"

    false_negatives_df = scored_validation_df[
        (scored_validation_df["label"] == 1) & (scored_validation_df["predicted_label"] == 0)
    ].copy()
    false_negatives_df["feedback_type"] = "false_negative"

    artifacts = load_index_artifacts(
        index_path=args.index_file,
        metadata_path=args.index_metadata_file,
        corpus_path=args.corpus_file,
        embeddings_path=args.embeddings_file,
    )
    hard_negatives_df = mine_hard_negatives(
        train_df=train_df,
        validation_df=validation_df,
        query_df=hard_negative_query_df,
        embedding_lookup=embedding_lookup,
        artifacts=artifacts,
        search_k=args.search_k,
        search_buffer=args.search_buffer,
        min_score=args.hard_negative_min_score,
    )

    feedback_frames = [
        false_positives_df[
            ["question1", "question1_id", "question2", "question2_id", "label", "similarity", "feedback_type"]
        ],
        false_negatives_df[
            ["question1", "question1_id", "question2", "question2_id", "label", "similarity", "feedback_type"]
        ],
        hard_negatives_df[
            ["question1", "question1_id", "question2", "question2_id", "label", "similarity", "feedback_type"]
        ],
    ]
    feedback_df = pd.concat(feedback_frames, ignore_index=True)
    feedback_df["source"] = "active_learning"

    updated_train_df = pd.concat(
        [
            train_df,
            feedback_df[["question1", "question1_id", "question2", "question2_id", "label"]]
            .assign(pair_id=lambda frame: [f"feedback_{idx:06d}" for idx in range(len(frame))]),
        ],
        ignore_index=True,
    ).drop_duplicates(subset=["question1", "question2", "label"])

    args.false_positives_output.parent.mkdir(parents=True, exist_ok=True)
    false_positives_df.to_csv(args.false_positives_output, index=False)
    false_negatives_df.to_csv(args.false_negatives_output, index=False)
    hard_negatives_df.to_csv(args.hard_negatives_output, index=False)
    feedback_df.to_csv(args.feedback_output, index=False)
    updated_train_df.to_csv(args.updated_train_output, index=False)

    summary = {
        "threshold": float(threshold),
        "false_positives": int(len(false_positives_df)),
        "false_negatives": int(len(false_negatives_df)),
        "hard_negative_anchor_file": str(hard_negative_anchor_file),
        "hard_negative_anchor_queries": int(len(hard_negative_query_df)),
        "hard_negative_positive_only": bool(args.hard_negative_positive_only),
        "hard_negative_use_both_question_columns": bool(args.hard_negative_use_both_question_columns),
        "hard_negatives": int(len(hard_negatives_df)),
        "feedback_examples": int(len(feedback_df)),
        "updated_train_rows": int(len(updated_train_df)),
    }

    print(f"Saved false positives to: {args.false_positives_output}")
    print(f"Saved false negatives to: {args.false_negatives_output}")
    print(f"Saved hard negatives to: {args.hard_negatives_output}")
    print(f"Saved feedback examples to: {args.feedback_output}")
    print(f"Saved updated training file to: {args.updated_train_output}")
    print(summary)


if __name__ == "__main__":
    main()

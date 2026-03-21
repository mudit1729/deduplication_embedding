"""Prepare Quora duplicate-question data for local experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

from utils import (
    DEFAULT_TRAIN_FILE,
    DEFAULT_VALIDATION_FILE,
    PROCESSED_DIR,
    RAW_DIR,
    ensure_project_dirs,
    minimal_normalize_text,
    save_json,
)

DEFAULT_DATASET_NAME = "sentence-transformers/quora-duplicates"
DEFAULT_SUBSET = "pair-class"
DEFAULT_RANDOM_STATE = 42


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--dataset_subset", default=DEFAULT_SUBSET)
    parser.add_argument("--train_size", type=int, default=30_000)
    parser.add_argument("--validation_size", type=int, default=5_000)
    parser.add_argument("--max_corpus_size", type=int, default=40_000)
    parser.add_argument("--random_state", type=int, default=DEFAULT_RANDOM_STATE)
    parser.add_argument(
        "--raw_output",
        type=Path,
        default=RAW_DIR / "quora_pair_class_sample_raw.csv",
    )
    parser.add_argument(
        "--train_output",
        type=Path,
        default=DEFAULT_TRAIN_FILE,
    )
    parser.add_argument(
        "--validation_output",
        type=Path,
        default=DEFAULT_VALIDATION_FILE,
    )
    parser.add_argument(
        "--corpus_output",
        type=Path,
        default=PROCESSED_DIR / "corpus.csv",
    )
    parser.add_argument(
        "--summary_output",
        type=Path,
        default=PROCESSED_DIR / "data_summary.json",
    )
    return parser.parse_args()


def sample_raw_rows(
    dataset_name: str,
    dataset_subset: str,
    total_rows: int,
    random_state: int,
) -> pd.DataFrame:
    """Load the Hugging Face dataset and stratified-sample rows."""
    dataset = load_dataset(dataset_name, dataset_subset, split="train")
    raw_df = dataset.to_pandas()[["sentence1", "sentence2", "label"]]
    raw_df = raw_df.dropna(subset=["sentence1", "sentence2", "label"]).copy()
    raw_df["label"] = raw_df["label"].astype(int)
    raw_df = raw_df[raw_df["label"].isin([0, 1])].reset_index(drop=True)

    if total_rows >= len(raw_df):
        return raw_df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    sampled_df, _ = train_test_split(
        raw_df,
        train_size=total_rows,
        stratify=raw_df["label"],
        random_state=random_state,
    )
    return sampled_df.reset_index(drop=True)


def clean_pairs(sampled_df: pd.DataFrame) -> pd.DataFrame:
    """Apply baseline cleaning while keeping the task realistic."""
    cleaned_df = sampled_df.rename(
        columns={"sentence1": "question1", "sentence2": "question2"}
    ).copy()

    for column in ("question1", "question2"):
        cleaned_df[column] = cleaned_df[column].map(minimal_normalize_text)

    cleaned_df = cleaned_df[
        (cleaned_df["question1"] != "") & (cleaned_df["question2"] != "")
    ].copy()
    cleaned_df["label"] = cleaned_df["label"].astype(int)
    return cleaned_df.reset_index(drop=True)


def build_corpus(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    max_corpus_size: int,
) -> pd.DataFrame:
    """Build a corpus of unique questions, always keeping validation coverage."""
    combined_df = pd.concat([train_df, validation_df], ignore_index=True)
    all_questions = pd.concat(
        [
            combined_df["question1"],
            combined_df["question2"],
        ],
        ignore_index=True,
    )
    frequencies = all_questions.value_counts()

    validation_questions = pd.Index(
        pd.concat(
            [validation_df["question1"], validation_df["question2"]],
            ignore_index=True,
        ).drop_duplicates()
    )
    selected_questions = validation_questions.tolist()
    selected_set = set(selected_questions)

    if max_corpus_size > 0 and len(selected_questions) < max_corpus_size:
        for question in frequencies.index.tolist():
            if question in selected_set:
                continue
            selected_questions.append(question)
            selected_set.add(question)
            if len(selected_questions) >= max_corpus_size:
                break
    elif max_corpus_size <= 0:
        selected_questions = frequencies.index.tolist()
        selected_set = set(selected_questions)

    corpus_df = pd.DataFrame({"question_text": selected_questions})
    corpus_df.insert(
        0,
        "question_id",
        [f"q_{idx:06d}" for idx in range(len(corpus_df))],
    )
    corpus_df["source_frequency"] = corpus_df["question_text"].map(frequencies).fillna(0).astype(int)
    corpus_df["in_train"] = corpus_df["question_text"].isin(
        set(train_df["question1"]).union(set(train_df["question2"]))
    )
    corpus_df["in_validation"] = corpus_df["question_text"].isin(
        set(validation_df["question1"]).union(set(validation_df["question2"]))
    )
    return corpus_df


def attach_question_ids(pairs_df: pd.DataFrame, corpus_df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    """Attach corpus question IDs where available."""
    text_to_id = dict(zip(corpus_df["question_text"], corpus_df["question_id"]))
    enriched_df = pairs_df.copy()
    enriched_df.insert(0, "pair_id", [f"{split_name}_{idx:06d}" for idx in range(len(enriched_df))])
    enriched_df["question1_id"] = enriched_df["question1"].map(text_to_id)
    enriched_df["question2_id"] = enriched_df["question2"].map(text_to_id)
    ordered_columns = [
        "pair_id",
        "question1_id",
        "question2_id",
        "question1",
        "question2",
        "label",
    ]
    return enriched_df[ordered_columns]


def main() -> None:
    """Run the data preparation pipeline."""
    args = parse_args()
    ensure_project_dirs()

    total_rows = args.train_size + args.validation_size
    sampled_raw_df = sample_raw_rows(
        dataset_name=args.dataset_name,
        dataset_subset=args.dataset_subset,
        total_rows=total_rows,
        random_state=args.random_state,
    )
    sampled_raw_df.to_csv(args.raw_output, index=False)

    cleaned_df = clean_pairs(sampled_raw_df)
    if len(cleaned_df) < total_rows:
        raise ValueError(
            "Too few rows remain after cleaning. "
            f"Needed {total_rows}, found {len(cleaned_df)}."
        )

    train_df, validation_df = train_test_split(
        cleaned_df,
        train_size=args.train_size,
        test_size=args.validation_size,
        stratify=cleaned_df["label"],
        random_state=args.random_state,
    )

    train_df = train_df.reset_index(drop=True)
    validation_df = validation_df.reset_index(drop=True)

    corpus_df = build_corpus(
        train_df=train_df,
        validation_df=validation_df,
        max_corpus_size=args.max_corpus_size,
    )

    train_output_df = attach_question_ids(train_df, corpus_df, split_name="train")
    validation_output_df = attach_question_ids(validation_df, corpus_df, split_name="validation")

    train_output_df.to_csv(args.train_output, index=False)
    validation_output_df.to_csv(args.validation_output, index=False)
    corpus_df.to_csv(args.corpus_output, index=False)

    summary = {
        "dataset_name": args.dataset_name,
        "dataset_subset": args.dataset_subset,
        "train_pairs": int(len(train_output_df)),
        "validation_pairs": int(len(validation_output_df)),
        "train_positive_rate": float(train_output_df["label"].mean()),
        "validation_positive_rate": float(validation_output_df["label"].mean()),
        "corpus_size": int(len(corpus_df)),
        "validation_question_coverage": float(
            validation_output_df["question1_id"].notna().mean()
            * validation_output_df["question2_id"].notna().mean()
        ),
    }
    save_json(summary, args.summary_output)

    print(f"Saved train pairs to: {args.train_output}")
    print(f"Saved validation pairs to: {args.validation_output}")
    print(f"Saved corpus to: {args.corpus_output}")
    print(f"Saved summary to: {args.summary_output}")
    print(summary)


if __name__ == "__main__":
    main()

"""Embed the corpus or fine-tune a sentence-transformer model."""

from __future__ import annotations

import argparse
from collections import defaultdict
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sentence_transformers import InputExample, SentenceTransformer, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from torch.utils.data import DataLoader

from utils import (
    DEFAULT_CORPUS_FILE,
    DEFAULT_EMBEDDING_METADATA_FILE,
    DEFAULT_EMBEDDINGS_FILE,
    DEFAULT_MODEL_NAME,
    DEFAULT_TRAIN_FILE,
    DEFAULT_VALIDATION_FILE,
    MODELS_DIR,
    detect_device,
    encode_texts,
    ensure_project_dirs,
    load_corpus,
    load_pairs,
    load_sentence_transformer,
    save_json,
    seed_everything,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["embed", "finetune"], default="embed")
    parser.add_argument("--model_name_or_path", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--device", default=None)

    parser.add_argument("--corpus_file", type=Path, default=DEFAULT_CORPUS_FILE)
    parser.add_argument("--output_embeddings", type=Path, default=DEFAULT_EMBEDDINGS_FILE)
    parser.add_argument(
        "--embedding_metadata_file",
        type=Path,
        default=DEFAULT_EMBEDDING_METADATA_FILE,
    )
    parser.add_argument("--embed_batch_size", type=int, default=128)

    parser.add_argument("--train_file", type=Path, default=DEFAULT_TRAIN_FILE)
    parser.add_argument("--validation_file", type=Path, default=DEFAULT_VALIDATION_FILE)
    parser.add_argument(
        "--output_model_dir",
        type=Path,
        default=MODELS_DIR / "quora_miniLM_finetuned",
    )
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--max_train_examples", type=int, default=30_000)
    parser.add_argument("--max_validation_examples", type=int, default=2_000)
    parser.add_argument(
        "--loss_type",
        choices=["cosine", "contrastive", "multiple_negatives", "triplet"],
        default="cosine",
    )
    parser.add_argument("--contrastive_margin", type=float, default=0.5)
    parser.add_argument(
        "--hard_negative_file",
        type=Path,
        default=None,
        help="Optional CSV of mined negatives for triplet training.",
    )
    parser.add_argument(
        "--triplets_per_positive",
        type=int,
        default=1,
        help="How many negatives to pair with each positive example in triplet mode.",
    )
    parser.add_argument(
        "--triplet_margin",
        type=float,
        default=0.2,
        help="Margin used for triplet loss. The default is tuned for cosine distance.",
    )
    parser.add_argument(
        "--triplet_distance_metric",
        choices=["cosine", "euclidean", "manhattan"],
        default="cosine",
    )
    parser.add_argument("--multiple_negatives_scale", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--embed_after_training", action="store_true")
    return parser.parse_args()


def select_subset(df: pd.DataFrame, max_examples: int, seed: int) -> pd.DataFrame:
    """Keep a stratified subset when requested."""
    if max_examples <= 0 or len(df) <= max_examples:
        return df.reset_index(drop=True)

    sample_counts = (
        df["label"]
        .value_counts(normalize=True)
        .mul(max_examples)
        .round()
        .astype(int)
    )

    # Keep counts valid and make the total exactly max_examples.
    for label, label_count in df["label"].value_counts().items():
        sample_counts.loc[label] = max(1, min(int(sample_counts.get(label, 0)), int(label_count)))

    difference = int(max_examples - sample_counts.sum())
    if difference != 0:
        ordered_labels = df["label"].value_counts().index.tolist()
        step = 1 if difference > 0 else -1
        while difference != 0:
            adjusted = False
            for label in ordered_labels:
                available = int((df["label"] == label).sum())
                proposed = int(sample_counts.loc[label] + step)
                if 1 <= proposed <= available:
                    sample_counts.loc[label] = proposed
                    difference -= step
                    adjusted = True
                    if difference == 0:
                        break
            if not adjusted:
                break

    sampled_frames = []
    for label, sample_size in sample_counts.items():
        group = df[df["label"] == label]
        sampled_frames.append(group.sample(n=int(sample_size), random_state=seed))

    return (
        pd.concat(sampled_frames, ignore_index=True)
        .sample(frac=1.0, random_state=seed)
        .reset_index(drop=True)
    )


def embed_corpus(
    corpus_file: Path,
    model_name_or_path: str,
    device: str | None,
    batch_size: int,
    output_embeddings: Path,
    metadata_file: Path,
    model: SentenceTransformer | None = None,
) -> None:
    """Encode and save the corpus embeddings."""
    corpus_df = load_corpus(corpus_file)
    if corpus_df.empty:
        raise ValueError("Corpus is empty. Run prepare_data.py first.")

    active_model = model or load_sentence_transformer(model_name_or_path, device)
    embeddings = encode_texts(
        active_model,
        corpus_df["question_text"].tolist(),
        batch_size=batch_size,
        show_progress_bar=True,
    )

    output_embeddings.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_embeddings, embeddings)

    metadata = {
        "corpus_file": str(corpus_file),
        "device": detect_device(device),
        "embedding_dim": int(embeddings.shape[1]),
        "embedding_rows": int(embeddings.shape[0]),
        "model_name_or_path": model_name_or_path,
        "normalized": True,
        "output_embeddings": str(output_embeddings),
    }
    save_json(metadata, metadata_file)

    print(f"Saved embeddings to: {output_embeddings}")
    print(f"Saved embedding metadata to: {metadata_file}")
    print(metadata)


def load_negative_source(path: Path) -> pd.DataFrame:
    """Load an optional mined-negative CSV."""
    if not path.exists():
        raise FileNotFoundError(f"Missing hard-negative file: {path}")

    negative_df = pd.read_csv(path)
    required_columns = {"question1", "question2"}
    missing_columns = required_columns.difference(negative_df.columns)
    if missing_columns:
        raise ValueError(
            f"Hard-negative file {path} is missing required columns: {sorted(missing_columns)}"
        )

    if "label" in negative_df.columns:
        negative_df = negative_df[negative_df["label"] == 0].copy()
    return negative_df.reset_index(drop=True)


def build_pair_examples(train_df: pd.DataFrame, loss_type: str) -> tuple[list[InputExample], dict[str, Any]]:
    """Build training examples for pairwise losses."""
    if loss_type == "multiple_negatives":
        positive_df = (
            train_df[train_df["label"] == 1]
            .drop_duplicates(subset=["question1", "question2"])
            .reset_index(drop=True)
        )
        if positive_df.empty:
            raise ValueError("MultipleNegativesRankingLoss requires positive pairs in the training file.")

        examples = [
            InputExample(texts=[row.question1, row.question2])
            for row in positive_df.itertuples(index=False)
        ]
        summary = {
            "training_example_type": "positive_pairs",
            "raw_train_rows": int(len(train_df)),
            "positive_pairs_used": int(len(examples)),
        }
        return examples, summary

    examples = [
        InputExample(
            texts=[row.question1, row.question2],
            label=float(row.label),
        )
        for row in train_df.itertuples(index=False)
    ]
    summary = {
        "training_example_type": "labeled_pairs",
        "raw_train_rows": int(len(train_df)),
        "pair_examples_used": int(len(examples)),
    }
    return examples, summary


def build_triplet_examples(
    train_df: pd.DataFrame,
    hard_negative_file: Path | None,
    triplets_per_positive: int,
) -> tuple[list[InputExample], dict[str, Any]]:
    """Build anchor-positive-negative triplets for retrieval-oriented training."""
    positive_df = train_df[train_df["label"] == 1].copy()
    negative_df = train_df[train_df["label"] == 0].copy()

    if positive_df.empty:
        raise ValueError("Triplet loss requires positive pairs in the training file.")
    if triplets_per_positive < 1:
        raise ValueError("--triplets_per_positive must be at least 1.")

    positive_lookup: dict[str, list[str]] = defaultdict(list)
    negative_lookup: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)

    for row in positive_df.itertuples(index=False):
        if row.question1 != row.question2:
            positive_lookup[row.question1].append(row.question2)
            positive_lookup[row.question2].append(row.question1)

    def register_negative(anchor: str, negative_text: str, score: float, source: str) -> None:
        if anchor == negative_text:
            return

        existing = negative_lookup[anchor].get(negative_text)
        if existing is None or score > float(existing["score"]):
            negative_lookup[anchor][negative_text] = {
                "score": float(score),
                "source": source,
            }

    for row in negative_df.itertuples(index=False):
        register_negative(row.question1, row.question2, 0.0, "train")
        register_negative(row.question2, row.question1, 0.0, "train")

    mined_negative_rows = 0
    if hard_negative_file is not None:
        mined_negative_df = load_negative_source(hard_negative_file)
        mined_negative_rows = int(len(mined_negative_df))
        similarity_present = "similarity" in mined_negative_df.columns
        for row in mined_negative_df.itertuples(index=False):
            score = float(getattr(row, "similarity", 1.0) if similarity_present else 1.0)
            register_negative(row.question1, row.question2, score, "mined")
            register_negative(row.question2, row.question1, score, "mined")

    triplet_examples: list[InputExample] = []
    anchors_with_negatives = 0
    mined_negative_matches = 0

    for anchor, positive_texts in positive_lookup.items():
        negative_candidates = negative_lookup.get(anchor, {})
        if not negative_candidates:
            continue

        unique_positives = list(dict.fromkeys(text for text in positive_texts if text != anchor))
        if not unique_positives:
            continue

        ordered_negatives = sorted(
            negative_candidates.items(),
            key=lambda item: (float(item[1]["score"]), item[1]["source"] == "mined"),
            reverse=True,
        )
        if not ordered_negatives:
            continue

        anchors_with_negatives += 1
        for positive_text in unique_positives:
            negatives_added = 0
            for negative_text, metadata in ordered_negatives:
                if negative_text == positive_text:
                    continue
                triplet_examples.append(
                    InputExample(texts=[anchor, positive_text, negative_text])
                )
                negatives_added += 1
                if metadata["source"] == "mined":
                    mined_negative_matches += 1
                if negatives_added >= triplets_per_positive:
                    break

    if not triplet_examples:
        raise ValueError(
            "No triplets could be built. Provide a training file with overlapping positive and "
            "negative anchors or pass a more compatible --hard_negative_file."
        )

    summary = {
        "training_example_type": "triplets",
        "raw_train_rows": int(len(train_df)),
        "positive_pairs_available": int(len(positive_df)),
        "train_negative_pairs_available": int(len(negative_df)),
        "mined_negative_rows": mined_negative_rows,
        "anchors_with_triplets": int(anchors_with_negatives),
        "triplet_examples_used": int(len(triplet_examples)),
        "mined_negative_triplets": int(mined_negative_matches),
        "triplets_per_positive": int(triplets_per_positive),
    }
    return triplet_examples, summary


def build_train_objective(
    args: argparse.Namespace,
    train_df: pd.DataFrame,
    model: SentenceTransformer,
) -> tuple[list[InputExample], Any, dict[str, Any]]:
    """Create training examples and the matching loss object."""
    if args.loss_type == "contrastive":
        examples, summary = build_pair_examples(train_df, loss_type=args.loss_type)
        train_loss = losses.ContrastiveLoss(
            model=model,
            margin=args.contrastive_margin,
        )
        return examples, train_loss, summary

    if args.loss_type == "multiple_negatives":
        examples, summary = build_pair_examples(train_df, loss_type=args.loss_type)
        train_loss = losses.MultipleNegativesRankingLoss(
            model=model,
            scale=args.multiple_negatives_scale,
        )
        return examples, train_loss, summary

    if args.loss_type == "triplet":
        examples, summary = build_triplet_examples(
            train_df=train_df,
            hard_negative_file=args.hard_negative_file,
            triplets_per_positive=args.triplets_per_positive,
        )
        distance_metric = {
            "cosine": losses.TripletDistanceMetric.COSINE,
            "euclidean": losses.TripletDistanceMetric.EUCLIDEAN,
            "manhattan": losses.TripletDistanceMetric.MANHATTAN,
        }[args.triplet_distance_metric]
        train_loss = losses.TripletLoss(
            model=model,
            distance_metric=distance_metric,
            triplet_margin=args.triplet_margin,
        )
        return examples, train_loss, summary

    examples, summary = build_pair_examples(train_df, loss_type=args.loss_type)
    train_loss = losses.CosineSimilarityLoss(model=model)
    return examples, train_loss, summary


def finetune_model(args: argparse.Namespace) -> SentenceTransformer:
    """Optionally fine-tune the baseline encoder on labeled pairs."""
    train_df = select_subset(load_pairs(args.train_file), args.max_train_examples, args.seed)
    validation_df = select_subset(
        load_pairs(args.validation_file), args.max_validation_examples, args.seed
    )

    if train_df.empty or validation_df.empty:
        raise ValueError("Training and validation data must be non-empty.")

    model = load_sentence_transformer(args.model_name_or_path, args.device)
    train_examples, train_loss, training_data_summary = build_train_objective(
        args=args,
        train_df=train_df,
        model=model,
    )

    effective_batch_size = min(args.train_batch_size, len(train_examples))
    if args.loss_type == "multiple_negatives" and effective_batch_size < 2:
        raise ValueError(
            "MultipleNegativesRankingLoss requires at least 2 positive pairs in a batch. "
            "Increase the training set size or reduce filtering."
        )

    train_dataloader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=effective_batch_size,
        drop_last=args.loss_type == "multiple_negatives",
    )

    evaluator = EmbeddingSimilarityEvaluator(
        sentences1=validation_df["question1"].tolist(),
        sentences2=validation_df["question2"].tolist(),
        scores=validation_df["label"].astype(float).tolist(),
        name="quora-validation",
    )

    warmup_steps = math.ceil(len(train_dataloader) * args.epochs * args.warmup_ratio)
    args.output_model_dir.mkdir(parents=True, exist_ok=True)
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=args.epochs,
        warmup_steps=warmup_steps,
        output_path=str(args.output_model_dir),
        show_progress_bar=True,
    )

    training_summary = {
        "base_model_name_or_path": args.model_name_or_path,
        "loss_type": args.loss_type,
        "contrastive_margin": args.contrastive_margin if args.loss_type == "contrastive" else None,
        "hard_negative_file": str(args.hard_negative_file) if args.hard_negative_file else None,
        "device": detect_device(args.device),
        "epochs": args.epochs,
        "multiple_negatives_scale": (
            args.multiple_negatives_scale if args.loss_type == "multiple_negatives" else None
        ),
        "train_batch_size": effective_batch_size,
        "max_train_examples": int(len(train_df)),
        "max_validation_examples": int(len(validation_df)),
        "output_model_dir": str(args.output_model_dir),
        "triplet_distance_metric": (
            args.triplet_distance_metric if args.loss_type == "triplet" else None
        ),
        "triplet_margin": args.triplet_margin if args.loss_type == "triplet" else None,
        "triplets_per_positive": (
            args.triplets_per_positive if args.loss_type == "triplet" else None
        ),
    }
    training_summary.update(training_data_summary)
    save_json(training_summary, args.output_model_dir / "training_summary.json")
    print(f"Saved fine-tuned model to: {args.output_model_dir}")
    print(training_summary)
    return SentenceTransformer(str(args.output_model_dir), device=detect_device(args.device))


def main() -> None:
    """Run embedding or optional fine-tuning."""
    args = parse_args()
    ensure_project_dirs()
    seed_everything(args.seed)

    if args.mode == "embed":
        embed_corpus(
            corpus_file=args.corpus_file,
            model_name_or_path=args.model_name_or_path,
            device=args.device,
            batch_size=args.embed_batch_size,
            output_embeddings=args.output_embeddings,
            metadata_file=args.embedding_metadata_file,
        )
        return

    model = finetune_model(args)
    if args.embed_after_training:
        embed_corpus(
            corpus_file=args.corpus_file,
            model_name_or_path=str(args.output_model_dir),
            device=args.device,
            batch_size=args.embed_batch_size,
            output_embeddings=args.output_embeddings,
            metadata_file=args.embedding_metadata_file,
            model=model,
        )


if __name__ == "__main__":
    main()

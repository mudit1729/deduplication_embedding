"""Embed the corpus or optionally fine-tune a sentence-transformer model."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

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
    parser.add_argument("--loss_type", choices=["cosine", "contrastive"], default="cosine")
    parser.add_argument("--contrastive_margin", type=float, default=0.5)
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


def finetune_model(args: argparse.Namespace) -> SentenceTransformer:
    """Optionally fine-tune the baseline encoder on labeled pairs."""
    train_df = select_subset(load_pairs(args.train_file), args.max_train_examples, args.seed)
    validation_df = select_subset(
        load_pairs(args.validation_file), args.max_validation_examples, args.seed
    )

    if train_df.empty or validation_df.empty:
        raise ValueError("Training and validation data must be non-empty.")

    model = load_sentence_transformer(args.model_name_or_path, args.device)
    train_examples = [
        InputExample(
            texts=[row.question1, row.question2],
            label=float(row.label),
        )
        for row in train_df.itertuples(index=False)
    ]

    train_dataloader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=args.train_batch_size,
    )
    if args.loss_type == "contrastive":
        train_loss = losses.ContrastiveLoss(
            model=model,
            margin=args.contrastive_margin,
        )
    else:
        train_loss = losses.CosineSimilarityLoss(model=model)

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
        "device": detect_device(args.device),
        "epochs": args.epochs,
        "train_batch_size": args.train_batch_size,
        "max_train_examples": int(len(train_df)),
        "max_validation_examples": int(len(validation_df)),
        "output_model_dir": str(args.output_model_dir),
    }
    save_json(training_summary, args.output_model_dir / "training_summary.json")
    print(f"Saved fine-tuned model to: {args.output_model_dir}")
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

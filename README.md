# Deduplication Embedding

Local near-duplicate question detection on the Quora Question Pairs dataset using:

- `sentence-transformers`
- `hnswlib`
- `datasets`
- `scikit-learn`
- `pandas`
- `numpy`

The repo is intentionally small enough to run on a MacBook, but it follows the same retrieval-first pattern you would use in production:

1. prepare a real labeled pair dataset
2. build a unique corpus of candidate questions
3. encode the corpus with a sentence-transformer
4. build a local ANN index with HNSW
5. retrieve top-k semantic neighbors
6. threshold cosine similarity for duplicate vs non-duplicate
7. evaluate both retrieval and pair classification
8. mine hard examples for a simple active-learning loop

## What Is In This Repo

```text
.
├── README.md
├── data/
│   ├── processed/
│   └── raw/
├── indices/
├── models/
├── notebooks/
├── requirements.txt
├── run_pipeline.sh
└── src/
    ├── active_learning.py
    ├── build_index.py
    ├── evaluate.py
    ├── prepare_data.py
    ├── search.py
    ├── train_encoder.py
    └── utils.py
```

## Dataset

- Source: [sentence-transformers/quora-duplicates](https://huggingface.co/datasets/sentence-transformers/quora-duplicates)
- Subset used here: `pair-class`
- Fields used: `sentence1`, `sentence2`, `label`

Default local subset sizes:

- train pairs: `30,000`
- validation pairs: `5,000`
- corpus size: `40,000` unique questions

## Mac Setup

Use Python `3.11` or `3.12`. The runs documented below were executed on Apple Silicon macOS with Python `3.11`.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

Notes:

- `accelerate` is included because current `sentence-transformers` fine-tuning uses the Hugging Face trainer stack.
- If `hnswlib` fails to build, run `xcode-select --install`.
- If MPS is unstable on your machine, add `--device cpu` to the training and evaluation commands.

## Quickstart

### 1. Prepare data

```bash
python src/prepare_data.py
```

Outputs:

- `data/raw/quora_pair_class_sample_raw.csv`
- `data/processed/train_pairs.csv`
- `data/processed/validation_pairs.csv`
- `data/processed/corpus.csv`
- `data/processed/data_summary.json`

### 2. Embed the baseline corpus

```bash
python src/train_encoder.py --mode embed
```

Default encoder:

- `sentence-transformers/all-MiniLM-L6-v2`

Outputs:

- `models/corpus_embeddings.npy`
- `models/embedding_metadata.json`

### 3. Build the ANN index

```bash
python src/build_index.py
```

Outputs:

- `indices/quora_hnsw.index`
- `indices/quora_hnsw_metadata.json`
- `indices/id_mapping.csv`

### 4. Evaluate

```bash
python src/evaluate.py
```

Metrics:

- `Recall@1`
- `Recall@5`
- `MRR`
- `Precision`
- `Recall`
- `F1`
- cosine threshold sweep
- average precision for pair scoring

Outputs:

- `models/validation_threshold_sweep.csv`
- `models/validation_pair_scores.csv`
- `models/validation_retrieval_results.csv`
- `models/evaluation_summary.json`

### 5. Search

```bash
python src/search.py --query "How can I learn Python fast?" --top_k 5
```

### 6. Mine active-learning examples

```bash
python src/active_learning.py
```

Outputs:

- `data/processed/false_positives.csv`
- `data/processed/false_negatives.csv`
- `data/processed/hard_negatives.csv`
- `data/processed/active_learning_feedback_examples.csv`
- `data/processed/train_pairs_active_learning.csv`

## Fine-Tuning

`src/train_encoder.py` supports two fine-tuning losses:

- `--loss_type cosine`
- `--loss_type contrastive`

The contrastive setup is the more useful one in this repo.

Example:

```bash
python src/train_encoder.py \
  --mode finetune \
  --loss_type contrastive \
  --contrastive_margin 0.5 \
  --epochs 1 \
  --train_batch_size 32 \
  --output_model_dir models/quora_miniLM_contrastive \
  --embed_after_training \
  --output_embeddings models/corpus_embeddings_contrastive.npy \
  --embedding_metadata_file models/embedding_metadata_contrastive.json
```

Then build and evaluate the matching index:

```bash
python src/build_index.py \
  --embeddings_file models/corpus_embeddings_contrastive.npy \
  --embedding_metadata_file models/embedding_metadata_contrastive.json \
  --index_file indices/quora_hnsw_contrastive.index \
  --metadata_file indices/quora_hnsw_contrastive_metadata.json \
  --mapping_file indices/id_mapping_contrastive.csv

python src/evaluate.py \
  --model_name_or_path models/quora_miniLM_contrastive \
  --embeddings_file models/corpus_embeddings_contrastive.npy \
  --index_file indices/quora_hnsw_contrastive.index \
  --index_metadata_file indices/quora_hnsw_contrastive_metadata.json \
  --threshold_csv models/validation_threshold_sweep_contrastive.csv \
  --pair_scores_csv models/validation_pair_scores_contrastive.csv \
  --retrieval_csv models/validation_retrieval_results_contrastive.csv \
  --summary_json models/evaluation_summary_contrastive.json
```

## Contrastive Loss -> Active Learning -> Retrain

This is the workflow that produced the best model in this repo.

### Step 1. Train a contrastive encoder

```bash
python src/train_encoder.py \
  --mode finetune \
  --loss_type contrastive \
  --contrastive_margin 0.5 \
  --epochs 1 \
  --train_batch_size 32 \
  --output_model_dir models/quora_miniLM_contrastive \
  --embed_after_training \
  --output_embeddings models/corpus_embeddings_contrastive.npy \
  --embedding_metadata_file models/embedding_metadata_contrastive.json
```

### Step 2. Build and evaluate the contrastive index

```bash
python src/build_index.py \
  --embeddings_file models/corpus_embeddings_contrastive.npy \
  --embedding_metadata_file models/embedding_metadata_contrastive.json \
  --index_file indices/quora_hnsw_contrastive.index \
  --metadata_file indices/quora_hnsw_contrastive_metadata.json \
  --mapping_file indices/id_mapping_contrastive.csv

python src/evaluate.py \
  --model_name_or_path models/quora_miniLM_contrastive \
  --embeddings_file models/corpus_embeddings_contrastive.npy \
  --index_file indices/quora_hnsw_contrastive.index \
  --index_metadata_file indices/quora_hnsw_contrastive_metadata.json \
  --threshold_csv models/validation_threshold_sweep_contrastive.csv \
  --pair_scores_csv models/validation_pair_scores_contrastive.csv \
  --retrieval_csv models/validation_retrieval_results_contrastive.csv \
  --summary_json models/evaluation_summary_contrastive.json
```

### Step 3. Mine active-learning examples from the contrastive model

```bash
python src/active_learning.py \
  --model_name_or_path models/quora_miniLM_contrastive \
  --embeddings_file models/corpus_embeddings_contrastive.npy \
  --index_file indices/quora_hnsw_contrastive.index \
  --index_metadata_file indices/quora_hnsw_contrastive_metadata.json \
  --threshold_summary_file models/evaluation_summary_contrastive.json \
  --false_positives_output data/processed/false_positives_contrastive.csv \
  --false_negatives_output data/processed/false_negatives_contrastive.csv \
  --hard_negatives_output data/processed/hard_negatives_contrastive.csv \
  --feedback_output data/processed/active_learning_feedback_examples_contrastive.csv \
  --updated_train_output data/processed/train_pairs_active_learning_contrastive.csv
```

What this produced in the documented run:

- feedback examples: `2,066`
- false positives: `527`
- false negatives: `244`
- hard negatives: `1,295`
- updated training rows: `31,539`

### Step 4. Retrain on the augmented file

```bash
python src/train_encoder.py \
  --mode finetune \
  --loss_type contrastive \
  --contrastive_margin 0.5 \
  --train_file data/processed/train_pairs_active_learning_contrastive.csv \
  --max_train_examples 0 \
  --epochs 1 \
  --train_batch_size 32 \
  --output_model_dir models/quora_miniLM_contrastive_active \
  --embed_after_training \
  --output_embeddings models/corpus_embeddings_contrastive_active.npy \
  --embedding_metadata_file models/embedding_metadata_contrastive_active.json
```

### Step 5. Build and evaluate the post-active-learning model

```bash
python src/build_index.py \
  --embeddings_file models/corpus_embeddings_contrastive_active.npy \
  --embedding_metadata_file models/embedding_metadata_contrastive_active.json \
  --index_file indices/quora_hnsw_contrastive_active.index \
  --metadata_file indices/quora_hnsw_contrastive_active_metadata.json \
  --mapping_file indices/id_mapping_contrastive_active.csv

python src/evaluate.py \
  --model_name_or_path models/quora_miniLM_contrastive_active \
  --embeddings_file models/corpus_embeddings_contrastive_active.npy \
  --index_file indices/quora_hnsw_contrastive_active.index \
  --index_metadata_file indices/quora_hnsw_contrastive_active_metadata.json \
  --threshold_csv models/validation_threshold_sweep_contrastive_active.csv \
  --pair_scores_csv models/validation_pair_scores_contrastive_active.csv \
  --retrieval_csv models/validation_retrieval_results_contrastive_active.csv \
  --summary_json models/evaluation_summary_contrastive_active.json
```

## Measured Results

The following metrics were produced from actual local runs in this repo on the default 30k / 5k / 40k setup.

| Model | Avg Precision | Precision | Recall | F1 | Best Threshold | Recall@1 | Recall@5 | MRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline MiniLM | 0.7628 | 0.6505 | 0.8451 | 0.7352 | 0.7600 | 0.6046 | 0.8267 | 0.6912 |
| Contrastive Fine-Tune | 0.8465 | 0.7525 | 0.8678 | 0.8060 | 0.7800 | 0.5997 | 0.8131 | 0.6824 |
| Contrastive + Active Learning | 0.8647 | 0.7714 | 0.9122 | 0.8359 | 0.7700 | 0.6051 | 0.8250 | 0.6892 |

Interpretation:

- contrastive loss gives a large win on pair classification quality
- a naive cosine-loss fine-tune helped classification too, but hurt retrieval more sharply, so it is not the focus here
- contrastive plus one active-learning retrain kept retrieval almost flat versus baseline while materially improving precision, recall, F1, and average precision

That final tradeoff is the best fit for this toy retrieval-first pipeline.

## Example Query

Using the final `contrastive + active learning` model:

```bash
python src/search.py \
  --model_name_or_path models/quora_miniLM_contrastive_active \
  --embeddings_file models/corpus_embeddings_contrastive_active.npy \
  --index_file indices/quora_hnsw_contrastive_active.index \
  --index_metadata_file indices/quora_hnsw_contrastive_active_metadata.json \
  --threshold_summary_file models/evaluation_summary_contrastive_active.json \
  --query "How can I learn Python fast?" \
  --top_k 5
```

Top result in the documented run:

- `What's the best way to learn Python?` with score `0.8759`

## One-Command Baseline Run

```bash
bash run_pipeline.sh
```

This script runs:

1. data preparation
2. baseline embedding
3. baseline HNSW build
4. baseline evaluation
5. a sample search

## Why This Mirrors Production

This repo is still a toy, but the system shape is realistic:

- dense embeddings for semantic candidate generation
- ANN retrieval for low-latency nearest-neighbor lookup
- a thresholding decision layer on top of retrieval
- offline evaluation for both retrieval and classification
- active-learning exports to drive the next training cycle

In a larger production system, you would usually:

- keep the corpus in a database or service instead of CSV
- refresh embeddings on a schedule
- use human or user feedback as labels
- route difficult cases into a stronger reranker or classifier

## Common Mac Issues

### Python version

- Avoid system Python `3.14` for this project
- Prefer `python3.11` or `python3.12`

### `hnswlib` build errors

```bash
xcode-select --install
python -m pip install --upgrade pip setuptools wheel
```

### MPS instability

Force CPU if needed:

```bash
python src/train_encoder.py --mode embed --device cpu
python src/evaluate.py --device cpu
python src/search.py --device cpu --query "How can I learn Python fast?"
```

### Slow first run

The first run downloads:

- the Quora dataset
- the MiniLM encoder
- tokenizer/model caches

After that, reruns are much faster.

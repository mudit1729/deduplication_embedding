#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

echo "[1/5] Preparing data"
python src/prepare_data.py

echo "[2/5] Generating corpus embeddings"
python src/train_encoder.py --mode embed

echo "[3/5] Building ANN index"
python src/build_index.py

echo "[4/5] Evaluating retrieval and classification"
python src/evaluate.py

echo "[5/5] Running a sample query"
python src/search.py --query "How can I learn Python fast?" --top_k 5

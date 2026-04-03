#!/usr/bin/env bash
set -euo pipefail

python3 data/cached_challenge_fineweb.py --variant sp1024

RUN_ID="$(date -u +%Y%m%d_%H%M%S)"
export RUN_ID

DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
MAX_WALLCLOCK_SECONDS=800 \
torchrun --standalone --nproc_per_node=1 train_gpt.py

mkdir -p "logs/${RUN_ID}"
mv final_model.pt "logs/${RUN_ID}/final_model.pt"
mv final_model.int8.ptz "logs/${RUN_ID}/final_model.int8.ptz"

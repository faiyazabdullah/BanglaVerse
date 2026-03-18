#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# All evaluation directories (languages + dialects)
EVAL_DIRS=(
  "language_eval/hindi_eval/eng_prompt/gpt5"
  "language_eval/en_eval/eng_prompt/gpt5"
  "language_eval/pure_bn_eval/eng_prompt/gpt5"
  "language_eval/urdu_eval/eng_prompt/gpt5"
  "dialect_eval/chittagong_eval/eng_prompt/gpt5"
  "dialect_eval/barishal_eval/eng_prompt/gpt5"
  "dialect_eval/noakhali_eval/eng_prompt/gpt5"
  "dialect_eval/rangpur_eval/eng_prompt/gpt5"
  "dialect_eval/sylhet_eval/eng_prompt/gpt5"
)

echo "=========================================="
echo " Running All Evaluations (Caption + VQA)"
echo "=========================================="

for dir in "${EVAL_DIRS[@]}"; do
  full_dir="$SCRIPT_DIR/$dir"

  # --- Caption ---
  caption_file="$full_dir/gpt_5_eval_captions.py"
  if [[ -f "$caption_file" ]]; then
    echo ""
    echo "------------------------------------------"
    echo " [CAPTION] $dir"
    echo "------------------------------------------"
    uv run "$caption_file"
  else
    echo " [SKIP] Caption file not found: $dir"
  fi

  # --- VQA ---
  vqa_file="$full_dir/gpt_5_vqa_eval.py"
  if [[ -f "$vqa_file" ]]; then
    echo ""
    echo "------------------------------------------"
    echo " [VQA] $dir"
    echo "------------------------------------------"
    uv run "$vqa_file"
  else
    echo " [SKIP] VQA file not found: $dir"
  fi
done

echo ""
echo "=========================================="
echo " All evaluations complete!"
echo "=========================================="

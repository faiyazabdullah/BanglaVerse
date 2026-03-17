#!/usr/bin/env python3
"""VQA evaluation for this language/dialect."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from tqdm import tqdm
from joblib import Parallel, delayed
from utils import (
    KEY_LIST, DATA_ROOT, RESULTS_ROOT, N_JOBS,
    process_vqa_sector,
)

# ---- Language-specific config ----

OUTPUT_ROOT = RESULTS_ROOT / "language_eval" / "urdu_eval" / "gpt5_vqa"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

SECTORS = {
    "culture": {
        "images": DATA_ROOT / "pure_bn" / "culture" / "images",
        "annotation": DATA_ROOT / "Other_languages" / "data_urdu" / "culture" / "annotations" / "culture_qa_pairs.json",
    },
    "food": {
        "images": DATA_ROOT / "pure_bn" / "food" / "images",
        "annotation": DATA_ROOT / "Other_languages" / "data_urdu" / "food" / "annotations" / "food_qa_pairs.json",
    },
    "history": {
        "images": DATA_ROOT / "pure_bn" / "history" / "images",
        "annotation": DATA_ROOT / "Other_languages" / "data_urdu" / "history" / "annotations" / "history_qa_pairs.json",
    },
    "media_and_movies": {
        "images": DATA_ROOT / "pure_bn" / "media_and_movies" / "images",
        "annotation": DATA_ROOT / "Other_languages" / "data_urdu" / "media_and_movies" / "annotations" / "media_and_movies_qa_pairs.json",
    },
    "national_achievements": {
        "images": DATA_ROOT / "pure_bn" / "national_achievements" / "images",
        "annotation": DATA_ROOT / "Other_languages" / "data_urdu" / "national_achievements" / "annotations" / "national_achievements_qa_pairs.json",
    },
    "nature": {
        "images": DATA_ROOT / "pure_bn" / "nature" / "images",
        "annotation": DATA_ROOT / "Other_languages" / "data_urdu" / "nature" / "annotations" / "nature_qa_pairs.json",
    },
    "personalities": {
        "images": DATA_ROOT / "pure_bn" / "personalities" / "images",
        "annotation": DATA_ROOT / "Other_languages" / "data_urdu" / "personalities" / "annotations" / "personalities_qa_pairs.json",
    },
    "politics": {
        "images": DATA_ROOT / "pure_bn" / "politics" / "images",
        "annotation": DATA_ROOT / "Other_languages" / "data_urdu" / "politics" / "annotations" / "politics_qa_pairs.json",
    },
    "sports": {
        "images": DATA_ROOT / "pure_bn" / "sports" / "images",
        "annotation": DATA_ROOT / "Other_languages" / "data_urdu" / "sports" / "annotations" / "sports_qa_pairs.json",
    },
}

# ---- Prompts ----

PROMPT_ZERO_SHOT = (
    "You are an AI assistant that answers visual multiple-choice questions in Urdu.\n"
    "Task:\n"
    "1. Look carefully at the given image.\n"
    "2. Read the question: {question}\n"
    "3. Review the provided answer choices: {options}\n"
    "4. Select the **single most accurate answer**.\n\n"
    "Response Rules:\n"
    "- The index must be the programming list index (starting from 0).\n"
    "- Respond ONLY with the exact format below.\n"
    "- Use Urdu text (Urdu script) for the answer option.\n"
    "- Do NOT add explanations, extra words, reasoning steps, or anything outside the specified format.\n"
    "- Follow this exact structure:\n\n"
    "Index: <option_index>, Answer: \"<option_text_in_Urdu>\""
)

PROMPT_FEW_SHOT_TEMPLATE = (
    "You are an AI assistant that answers visual multiple-choice questions in Urdu.\n"
    "Task:\n"
    "1. Look carefully at the given image.\n"
    "2. Read the question: {question}\n"
    "3. Review the provided answer choices: {options}\n"
    "4. Select the **single most accurate answer**.\n\n"
    "Response Rules:\n"
    "- The index must be the programming list index (starting from 0).\n"
    "- Respond ONLY with the exact format below.\n"
    "- Use Urdu text (Urdu script) for the answer option.\n"
    "- Do NOT add explanations, extra words, reasoning steps, or anything outside the specified format.\n"
    "- Follow this exact structure:\n"
    "Index: <option_index>, Answer: \"<option_text_in_Urdu>\"\n\n"
    "Examples:\n{examples}\n"
    "Now, answer for the given image."
)

PROMPT_CHAIN_OF_THOUGHTS = (
    "You are an AI assistant that answers visual multiple-choice questions in Urdu.\n\n"
    "Task:\n"
    "1. Look carefully at the given image.\n"
    "2. Read the question: {question}\n"
    "3. Review the provided answer choices: {options}\n"
    "4. Select the **single most accurate answer**.\n\n"
    "Response Rules:\n"
    "- The index must be the programming list index (starting from 0).\n"
    "- Use Urdu text (Urdu script) for the answer option.\n"
    "- In Reasoning_En, write step-by-step reasoning in English — break down the solution logically:\n"
    "  Step 1: Describe key visual observations.\n"
    "  Step 2: Match observations to relevant answer choices.\n"
    "  Step 3: Eliminate incorrect choices with brief justification.\n"
    "  Step 4: Conclude why the final choice is correct.\n"
    "- Be clear, concise, and factual (avoid overly long explanations).\n"
    "- Follow this exact response format:\n\n"
    "Reasoning_En:\n"
    "Step 1: <your_observations>\n"
    "Step 2: <your_matching_logic>\n"
    "Step 3: <your_elimination_of_wrong_options>\n"
    "Step 4: <your_final_choice_reasoning>\n\n"
    "Final Answer: Index: <option_index>, Answer: \"<option_text_in_Urdu>\""
)

# ---- Main ----

def _run_vqa(sector_name, cfg, prompt_mode):
    try:
        process_vqa_sector(
            sector_name, cfg, KEY_LIST, OUTPUT_ROOT,
            PROMPT_ZERO_SHOT, PROMPT_FEW_SHOT_TEMPLATE, PROMPT_CHAIN_OF_THOUGHTS,
            prompt_mode=prompt_mode,
        )
    except Exception as e:
        print(f"Error processing sector {sector_name} ({prompt_mode}-shot): {e}")


def main():
    jobs = [
        (sector_name, cfg, prompt_mode)
        for prompt_mode in ["zero", "few", "cot"]
        for sector_name, cfg in SECTORS.items()
    ]
    print(f"Running {len(jobs)} VQA jobs with {N_JOBS} workers")
    Parallel(n_jobs=N_JOBS, backend="threading")(
        delayed(_run_vqa)(s, c, m) for s, c, m in tqdm(jobs, desc="VQA jobs")
    )


if __name__ == "__main__":
    main()

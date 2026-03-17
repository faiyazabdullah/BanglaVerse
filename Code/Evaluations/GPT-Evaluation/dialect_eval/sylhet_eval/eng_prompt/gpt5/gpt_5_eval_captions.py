#!/usr/bin/env python3
"""Caption evaluation for this language/dialect."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from tqdm import tqdm
from joblib import Parallel, delayed
from utils import (
    KEY_LIST, DATA_ROOT, RESULTS_ROOT, N_JOBS,
    process_caption_sector,
)

# ---- Language-specific config ----

PROMPT_ZERO_SHOT = (
    "You are an assistant that generates short, fluent captions in Sylhet dialect only. "
    "Look carefully at the given image and write exactly one meaningful sentence describing it. "
    "Do not use any English words, do not add extra explanations, labels, or quotes. "
    "Your entire output must be only the Sylhet dialect caption as plain text."
)

PROMPT_FEW_SHOT_TEMPLATE = (
    "You are an assistant that generates short, fluent captions in Sylhet dialect only.\n\n"
    "Examples:\n{examples}\n"
    "Now, generate a caption for the following image. "
    "Write exactly one meaningful Sylhet dialect sentence. "
    "Do not use any English words, do not add extra explanations, labels, or quotes. "
    "Your entire output must be only the Sylhet dialect caption as plain text."
)

OUTPUT_ROOT = RESULTS_ROOT / "dialect_eval" / "sylhet_eval" / "gpt5_captions"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

SECTORS = {
    "culture": {
        "images": DATA_ROOT / "pure_bn" / "culture" / "images",
        "annotation": DATA_ROOT / "dialectual_data" / "Sylhet" / "Results" / "Captions" / "culture_captions_sylhet.json",
    },
    "food": {
        "images": DATA_ROOT / "pure_bn" / "food" / "images",
        "annotation": DATA_ROOT / "dialectual_data" / "Sylhet" / "Results" / "Captions" / "food_captions_sylhet.json",
    },
    "history": {
        "images": DATA_ROOT / "pure_bn" / "history" / "images",
        "annotation": DATA_ROOT / "dialectual_data" / "Sylhet" / "Results" / "Captions" / "history_captions_sylhet.json",
    },
    "media_and_movies": {
        "images": DATA_ROOT / "pure_bn" / "media_and_movies" / "images",
        "annotation": DATA_ROOT / "dialectual_data" / "Sylhet" / "Results" / "Captions" / "media_and_movies_captions_sylhet.json",
    },
    "national_achievements": {
        "images": DATA_ROOT / "pure_bn" / "national_achievements" / "images",
        "annotation": DATA_ROOT / "dialectual_data" / "Sylhet" / "Results" / "Captions" / "national_achievements_captions_sylhet.json",
    },
    "nature": {
        "images": DATA_ROOT / "pure_bn" / "nature" / "images",
        "annotation": DATA_ROOT / "dialectual_data" / "Sylhet" / "Results" / "Captions" / "nature_captions_sylhet.json",
    },
    "personalities": {
        "images": DATA_ROOT / "pure_bn" / "personalities" / "images",
        "annotation": DATA_ROOT / "dialectual_data" / "Sylhet" / "Results" / "Captions" / "personalities_captions_sylhet.json",
    },
    "politics": {
        "images": DATA_ROOT / "pure_bn" / "politics" / "images",
        "annotation": DATA_ROOT / "dialectual_data" / "Sylhet" / "Results" / "Captions" / "politics_captions_sylhet.json",
    },
    "sports": {
        "images": DATA_ROOT / "pure_bn" / "sports" / "images",
        "annotation": DATA_ROOT / "dialectual_data" / "Sylhet" / "Results" / "Captions" / "sports_captions_sylhet.json",
    },
}

BERTSCORE_LANG = "bn"

# ---- Main ----

def _run_caption(sector_name, cfg, prompt_mode):
    try:
        process_caption_sector(
            sector_name, cfg, KEY_LIST, OUTPUT_ROOT,
            PROMPT_ZERO_SHOT, PROMPT_FEW_SHOT_TEMPLATE,
            bertscore_lang=BERTSCORE_LANG,
            prompt_mode=prompt_mode,
        )
    except Exception as e:
        print(f"Error processing sector {sector_name} ({prompt_mode}-shot): {e}")


def main():
    jobs = [
        (sector_name, cfg, prompt_mode)
        for prompt_mode in ["zero", "few"]
        for sector_name, cfg in SECTORS.items()
    ]
    print(f"Running {len(jobs)} caption jobs with {N_JOBS} workers")
    Parallel(n_jobs=N_JOBS, backend="threading")(
        delayed(_run_caption)(s, c, m) for s, c, m in tqdm(jobs, desc="Caption jobs")
    )


if __name__ == "__main__":
    main()

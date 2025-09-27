#!/usr/bin/env python3
"""
gemini_csu_rotate_keys_run_both_modes.py

- Rotates through multiple Gemini API keys on failure (quota/429/etc).
- Retries the same example after switching keys (doesn't skip or restart earlier items).
- Persists per-sector results so runs can be resumed without redoing answered items.
- Computes Exact Match (%) and BERTScore_F1 (with rescale fallback).
- Runs both zero-shot and few-shot modes for all sectors (outputs separate files per mode).
"""

import os
import json
import time
from pathlib import Path
from tqdm import tqdm
import google.generativeai as genai
from bert_score import score as bertscore_score

# ---------------- CONFIG (KEY LIST) ----------------
KEY_LIST = [
    # Add your Gemini API keys here
]

MODEL = "gemini-2.0-flash"
IMG_EXTS = {".png", ".jpg", ".jpeg"}

OUTPUT_ROOT = Path(
    r"...\gemini_2.0_csu"
)
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# --- SECTORS (unchanged) ---
SECTORS = {
    "culture": {
        "images": Path(
            r"...\data\culture\images"
        ),
        "annotation": Path(
            r"...\data\culture\annotations\culture_commonsense_reasoning.json"
        ),
    },
    "food": {
        "images": Path(
            r"...\data\food\images"
        ),
        "annotation": Path(
            r"...\data\food\annotations\food_commonsense_reasoning.json"
        ),
    },
    "history": {
        "images": Path(
            r"...\data\history\images"
        ),
        "annotation": Path(
            r"...\data\history\annotations\history_commonsense_reasoning.json"
        ),
    },
    "media_and_movies": {
        "images": Path(
            r"...\data\media_and_movies\images"
        ),
        "annotation": Path(
            r"...\data\media_and_movies\annotations\media_and_movies_commonsense_reasoning.json"
        ),
    },
    "national_achievements": {
        "images": Path(
            r"...\data\national_achievements\images"
        ),
        "annotation": Path(
            r"...\data\national_achievements\annotations\national_achievements_commonsense_reasoning.json"
        ),
    },
    "nature": {
        "images": Path(
            r"...\data\nature\images"
        ),
        "annotation": Path(
            r"...\data\nature\annotations\nature_commonsense_reasoning.json"
        ),
    },
    "personalities": {
        "images": Path(
            r"...\data\personalities\images"
        ),
        "annotation": Path(
            r"...\data\personalities\annotations\personalities_commonsense_reasoning.json"
        ),
    },
    "politics": {
        "images": Path(
            r"...\data\politics\images"
        ),
        "annotation": Path(
            r"...\data\politics\annotations\politics_commonsense_reasoning.json"
        ),
    },
    "sports": {
        "images": Path(
            r"...\data\sports\images"
        ),
        "annotation": Path(
            r"...\data\sports\annotations\sports_commonsense_reasoning.json"
        ),
    },
}

PROMPT_ZERO_SHOT = (
    "You are an expert assistant for Bangla culture and commonsense reasoning tasks. "
    "You are given an image and a question. Carefully look at the image and answer the question.\n\n"
    "RESPONSE RULES (VERY IMPORTANT):\n"
    "1) Your answer MUST be written **only in Bangla** (Bangla script).\n"
    "2) Use Bangla digits for numbers if possible.\n"
    "3) Keep the answer short, direct, and on a single line (no extra explanation).\n"
    "4) Do NOT include any English words, labels, quotes, explanations, or metadata.\n"
    "5) If you are not sure about the answer, respond exactly with: অনুমান করা যাচ্ছে না\n\n"
    "Image: {image_path}\n"
    "Question: {question}\n"
    "Answer:"
)

PROMPT_FEW_SHOT = (
    "You are an expert assistant for Bangla culture and commonsense reasoning tasks. "
    "You are given an image and a question. Carefully look at the image and answer the question.\n\n"
    "RESPONSE RULES (VERY IMPORTANT):\n"
    "1) Your answer MUST be written **only in Bangla** (Bangla script).\n"
    "2) Use Bangla digits for numbers if possible.\n"
    "3) Keep the answer short, direct, and on a single line (no extra explanation).\n"
    "4) Do NOT include any English words, labels, quotes, explanations, or metadata.\n"
    "5) If you are not sure about the answer, respond exactly with: অনুমান করা যাচ্ছে না\n\n"
    "EXAMPLES:\n"
    "Image: ./dataset/history/images/history_002.png\n"
    "Question: \"ছবিতে কী হচ্ছে?\"\n"
    "Answer: বাংলাদেশ মুক্তিযুদ্ধের আত্মসমর্পণ দৃশ্য\n\n"
    "Image: ./dataset/culture/images/culture_003.png\n"
    "Question: \"ছবির এই দিনে মানুষ কেন ফুল দিয়ে সাজে?\"\n"
    "Answer: পহেলা ফাল্গুন উদযাপন\n\n"
    "Image: ./dataset/sports/images/sports_001.png\n"
    "Question: \"ছবিতে সাকিব আল হাসান কিসের জন্য বিখ্যাত?\"\n"
    "Answer: অলরাউন্ডার হিসেবে খ্যাত\n\n"
    "Now, answer for the given image strictly following the above rules.\n\n"
    "Image: {image_path}\n"
    "Question: {question}\n"
    "Answer:"
)


def normalize_text(text):
    import unicodedata, re
    if text is None:
        return ""
    t = unicodedata.normalize("NFKC", str(text))
    t = re.sub(r"\s+", " ", t).strip()
    return t


# ----------------- Rotating Gemini CSU client -----------------
class RotatingGeminiCSU:
    def __init__(self, key_list, model_name):
        assert key_list, "Provide at least one API key"
        self.keys = key_list
        self.model_name = model_name
        self.key_index = 0
        self._configure_current_key()

    def _configure_current_key(self):
        key = self.keys[self.key_index]
        genai.configure(api_key=key)
        print(f"➡️ Using API key index {self.key_index}")

    def _advance_key(self):
        old = self.key_index
        self.key_index = (self.key_index + 1) % len(self.keys)
        print(f"🔁 Switching API key: {old} -> {self.key_index}")
        self._configure_current_key()

    def ask_once(self, image_path, question, prompt_mode="few"):
        """One API attempt with currently configured key. Exceptions propagate."""
        if prompt_mode == "zero":
            prompt = PROMPT_ZERO_SHOT.format(image_path=str(image_path), question=question)
        else:
            # use the few-shot prompt template and format in the image+question
            prompt = PROMPT_FEW_SHOT.format(image_path=str(image_path), question=question)

        img = genai.upload_file(str(image_path))
        model = genai.GenerativeModel(self.model_name)
        resp = model.generate_content([prompt, img])
        if hasattr(resp, "text") and resp.text:
            return resp.text.strip()
        elif hasattr(resp, "candidates") and resp.candidates:
            cand_text = resp.candidates[0].content.parts[0].text
            return cand_text.strip()
        else:
            return "⚠️ Empty response"

    def ask_with_rotation(self, image_path, question, prompt_mode="few", max_attempts_per_example=10, backoff_base=1.0):
        """
        Try up to max_attempts_per_example (across keys). On quota/429 or other errors,
        rotate to the next key and retry the same example.
        """
        attempt = 0
        last_exc = None
        while attempt < max_attempts_per_example:
            attempt += 1
            try:
                return self.ask_once(image_path, question, prompt_mode=prompt_mode)
            except Exception as e:
                last_exc = e
                msg = str(e).lower()
                if "429" in msg or "quota" in msg or "rate limit" in msg or "quota exceeded" in msg:
                    print(f"❗ Quota/rate-limit detected on key index {self.key_index}: {e}")
                    self._advance_key()
                    time.sleep(1.0)
                    continue
                # transient/other errors: rotate and backoff
                print(f"❗ API call failed on key index {self.key_index} (attempt {attempt}/{max_attempts_per_example}): {e}")
                self._advance_key()
                sleep_time = backoff_base * (2 ** (attempt - 1))
                print(f"⏳ backing off for {sleep_time:.1f}s before retrying this example")
                time.sleep(min(sleep_time, 60))
                continue

        print(f"❌ Failed to answer for {image_path.name} after {max_attempts_per_example} attempts. Last error: {last_exc}")
        return "❌ Failed to answer"


# ----------------- I/O helpers -----------------
def load_csu_annotations(path: Path):
    if not path.exists():
        print(f"⚠️ Annotation file not found: {path}")
        return []
    with open(path, "r", encoding="utf8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else []


def list_images(folder: Path):
    if not folder.exists():
        print(f"⚠️ Image folder not found: {folder}")
        return []
    return sorted([p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS])


def load_existing_results(out_file: Path):
    if out_file.exists():
        try:
            with open(out_file, "r", encoding="utf8") as f:
                data = json.load(f)
            # data: list of result dicts with image_id
            processed_ids = {item["image_id"] for item in data if "image_id" in item}
            return data, processed_ids
        except Exception as e:
            print(f"⚠️ Could not load existing results {out_file}: {e}")
            return [], set()
    return [], set()


def save_results_atomic(out_file: Path, results_list):
    tmp = out_file.with_suffix(out_file.suffix + ".tmp")
    with open(tmp, "w", encoding="utf8") as f:
        json.dump(results_list, f, ensure_ascii=False, indent=2)
    tmp.replace(out_file)


# ---------- metrics (Exact Match + BERTScore) ----------
def compute_metrics(preds, refs):
    # Exact Match %
    em_scores = [int(normalize_text(p) == normalize_text(r)) for p, r in zip(preds, refs)]
    em = round(100.0 * sum(em_scores) / len(em_scores), 2) if em_scores else 0.0

    # BERTScore (try rescaled; fallback to raw)
    bertscore = None
    try:
        P, R, F1 = bertscore_score(preds, refs, lang="bn", rescale_with_baseline=True)
        bertscore = float(F1.mean())
    except Exception as e:
        print(f"⚠️ BERTScore rescale failed: {e}. Falling back to raw BERTScore (rescale_with_baseline=False).")
        try:
            P, R, F1 = bertscore_score(preds, refs, lang="bn", rescale_with_baseline=False)
            bertscore = float(F1.mean())
        except Exception as e2:
            print(f"❗ BERTScore computation failed entirely: {e2}")
            bertscore = None

    return {
        "Exact Match (%)": em,
        "BERTScore_F1": bertscore,
        "n_examples": len(preds),
    }


# ----------------- Processing logic with resume support -----------------
def process_sector(sector_name, sector_cfg, key_list, prompt_mode="few", n_examples=10):
    print(f"\n==== Processing sector: {sector_name} (prompt_mode={prompt_mode}) ====")
    images_dir = sector_cfg["images"]
    annotation_file = sector_cfg["annotation"]

    csu_data = load_csu_annotations(annotation_file)
    csu_data = csu_data[:n_examples]  # Only process first n_examples examples

    if not csu_data:
        print("⚠️ No CSU data for this sector.")
        return

    out_file = OUTPUT_ROOT / f"{sector_name}_csu_gemini_{prompt_mode}.json"
    results_list, processed_ids = load_existing_results(out_file)

    client = RotatingGeminiCSU(key_list, MODEL)
    preds, refs = [], []

    # Build ordered list of examples to process (preserve original ordering)
    examples_to_process = [item for item in csu_data if item.get("image_id") not in processed_ids]
    print(f"Total examples: {len(csu_data)}; already done: {len(processed_ids)}; to process: {len(examples_to_process)}")

    if not examples_to_process:
        print("✅ Nothing to process; computing metrics from existing results.")
        # load preds/refs from existing results
        for r in results_list:
            preds.append(r.get("predicted_answer", ""))
            refs.append(r.get("ground_truth_answer", ""))
        metrics = compute_metrics(preds, refs) if refs else {"error": "no_examples"}
        metrics_out = OUTPUT_ROOT / f"{sector_name}_csu_metrics_gemini_{prompt_mode}.json"
        with open(metrics_out, "w", encoding="utf8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        return

    for item in tqdm(examples_to_process, desc=f"CSU ({sector_name})"):
        image_id = item.get("image_id")
        question = item.get("question")
        answer = item.get("answer")  # Reference answer

        # locate image (png, jpg, jpeg)
        img_path = None
        for ext in (".png", ".jpg", ".jpeg"):
            candidate = images_dir / f"{image_id}{ext}"
            if candidate.exists():
                img_path = candidate
                break
        if img_path is None:
            print(f"⚠️ Image not found for {image_id}")
            continue

        # ask with rotation (this will retry across keys if necessary)
        pred_answer = client.ask_with_rotation(img_path, question, prompt_mode=prompt_mode, max_attempts_per_example=10)
        preds.append(pred_answer)
        refs.append(answer)
        results_list.append(
            {
                "image_id": image_id,
                "question": question,
                "ground_truth_answer": answer,
                "predicted_answer": pred_answer,
            }
        )

        # persist after each example so we can resume later
        save_results_atomic(out_file, results_list)

        # be polite — keep a small pause between examples (adjust as needed)
        time.sleep(5)

    # compute metrics and save
    metrics = compute_metrics(preds, refs)
    metrics_out = OUTPUT_ROOT / f"{sector_name}_csu_metrics_gemini_{prompt_mode}.json"
    with open(metrics_out, "w", encoding="utf8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"✅ Finished sector {sector_name}. Results: {out_file}, Metrics: {metrics_out}")


def main():
    # Run for both zero-shot and few-shot modes for all sectors
    for prompt_mode in ["zero", "few"]:
        print(f"\n########### Starting run for prompt_mode={prompt_mode} ###########")
        for sector_name, cfg in SECTORS.items():
            try:
                process_sector(sector_name, cfg, KEY_LIST, prompt_mode=prompt_mode, n_examples=10)
            except Exception as e:
                print(f"❗ Error processing sector {sector_name} ({prompt_mode}-shot): {e}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
gemini_vqa_rotate_keys_run_both_modes.py

- Rotates through multiple Gemini API keys on failure (quota/429/etc).
- Retries the same example after switching keys (doesn't skip or restart earlier items).
- Persists per-sector results so runs can be resumed without redoing answered items.
- Computes Accuracy (%) for the multiple-choice VQA tasks.
- Runs both zero-shot and few-shot modes for all sectors (outputs separate files per mode).
"""

import os
import json
import time
from pathlib import Path
from tqdm import tqdm
import google.generativeai as genai

# ---------------- CONFIG (KEY LIST) ----------------
KEY_LIST = [
    # Add your Gemini API keys here
]

MODEL = "gemini-2.5-flash"
IMG_EXTS = {".png", ".jpg", ".jpeg"}

OUTPUT_ROOT = Path(r"...\gemini_2.5_vqa")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

SECTORS = {
    "culture": {
        "images": Path(r"...\data\culture\images"),
        "annotation": Path(r"...\data\culture\annotations\culture_qa_pairs.json")
    },
    "food": {
        "images": Path(r"...\data\food\images"),
        "annotation": Path(r"...\data\food\annotations\food_qa_pairs.json")
    },
    "history": {
        "images": Path(r"...\data\history\images"),
        "annotation": Path(r"...\data\history\annotations\history_qa_pairs.json")
    },
    "media_and_movies": {
        "images": Path(r"...\data\media_and_movies\images"),
        "annotation": Path(r"...\data\media_and_movies\annotations\media_and_movies_qa_pairs.json")
    },
    "national_achievements": {
        "images": Path(r"...\data\national_achievements\images"),
        "annotation": Path(r"...\data\national_achievements\annotations\national_achievements_qa_pairs.json")
    },
    "nature": {
        "images": Path(r"...\data\nature\images"),
        "annotation": Path(r"...\data\nature\annotations\nature_qa_pairs.json")
    },
    "personalities": {
        "images": Path(r"...\data\personalities\images"),
        "annotation": Path(r"...\data\personalities\annotations\personalities_qa_pairs.json")
    },
    "politics": {
        "images": Path(r"...\data\politics\images"),
        "annotation": Path(r"...\data\politics\annotations\politics_qa_pairs.json")
    },
    "sports": {
        "images": Path(r"...\data\sports\images"),
        "annotation": Path(r"...\data\sports\annotations\sports_qa_pairs.json")
    }
}

PROMPT_ZERO_SHOT = (
    "You are an AI assistant that answers visual multiple-choice questions in Bangla.\n"
    "Task:\n"
    "1. Look carefully at the given image: {image_path}\n"
    "2. Read the question: {question}\n"
    "3. Review the provided answer choices: {options}\n"
    "4. Select the **single most accurate answer**.\n\n"
    "Response Rules:\n"
    "- The index must be the programming list index (starting from 0).\n"
    "- Respond ONLY with the exact format below.\n"
    "- Use Bangla text for the answer option.\n"
    "- Do NOT add explanations, extra words, reasoning steps, or anything outside the specified format.\n"
    "- Follow this exact structure:\n\n"
    "Index: <option_index>, Answer: \"<option_text_in_Bangla>\""
)


PROMPT_FEW_SHOT = (
    "You are an AI assistant that answers visual multiple-choice questions in Bangla.\n"
    "Task:\n"
    "1. Look carefully at the given image: {image_path}\n"
    "2. Read the question: {question}\n"
    "3. Review the provided answer choices: {options}\n"
    "4. Select the **single most accurate answer**.\n\n"
    "Response Rules:\n"
    "- The index must be the programming list index (starting from 0).\n"
    "- Respond ONLY with the exact format below.\n"
    "- Use Bangla text for the answer option.\n"
    "- Do NOT add explanations, extra words, reasoning steps, or anything outside the specified format.\n"
    "- Follow this exact structure:\n"
    "Index: <option_index>, Answer: \"<option_text_in_Bangla>\"\n\n"
    "Examples:\n"
    "Image: ./dataset/culture/images/culture_003.png\n"
    "Question: \"ছবিতে কোন উৎসব পালিত হচ্ছে?\"\n"
    "Options: [\"পহেলা বৈশাখ\", \"ঈদ\", \"নববর্ষ\", \"নবন্ন\"]\n"
    "Answer: Index: 0, Answer: \"পহেলা বৈশাখ\"\n\n"
    "Image: ./dataset/history/images/history_002.png\n"
    "Question: \"এই ঘটনার পর বাংলাদেশে কী পরিবর্তন ঘটে?\"\n"
    "Options: [\"সংবিধান প্রণয়ন\", \"স্বাধীনতা অর্জন\", \"সেনা শাসনের শুরু\", \"গণভোট\"]\n"
    "Answer: Index: 1, Answer: \"স্বাধীনতা অর্জন\"\n\n"
    "Image: ./dataset/sports/images/sports_001.png\n"
    "Question: \"এই ক্রিকেটারটি কে?\"\n"
    "Options: [\"মাশরাফি বিন মোর্ত্তজা\", \"সাকিব আল হাসান\", \"মুশফিকুর রহিম\", \"তামিম ইকবাল\"]\n"
    "Answer: Index: 1, Answer: \"সাকিব আল হাসান\"\n\n"
    "Now, answer for the given image."
)



def normalize_text(text):
    import unicodedata, re
    if text is None:
        return ""
    t = unicodedata.normalize("NFKC", str(text))
    t = re.sub(r"\s+", " ", t).strip()
    return t


# ----------------- Rotating Gemini VQA client -----------------
class RotatingGeminiVQA:
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

    def ask_once(self, image_path, question, options, prompt_mode="few"):
        if prompt_mode == "zero":
            prompt = PROMPT_ZERO_SHOT.format(image_path=str(image_path), question=question, options=json.dumps(options, ensure_ascii=False))
        else:
            prompt = PROMPT_FEW_SHOT.format(image_path=str(image_path), question=question, options=json.dumps(options, ensure_ascii=False))

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

    def ask_with_rotation(self, image_path, question, options, prompt_mode="few", max_attempts_per_example=10, backoff_base=1.0):
        attempt = 0
        last_exc = None
        while attempt < max_attempts_per_example:
            attempt += 1
            try:
                return self.ask_once(image_path, question, options, prompt_mode=prompt_mode)
            except Exception as e:
                last_exc = e
                msg = str(e).lower()
                if "429" in msg or "quota" in msg or "rate limit" in msg or "quota exceeded" in msg:
                    print(f"❗ Quota/rate-limit detected on key index {self.key_index}: {e}")
                    self._advance_key()
                    time.sleep(1.0)
                    continue
                print(f"❗ API call failed on key index {self.key_index} (attempt {attempt}/{max_attempts_per_example}): {e}")
                self._advance_key()
                sleep_time = backoff_base * (2 ** (attempt - 1))
                print(f"⏳ backing off for {sleep_time:.1f}s before retrying this example")
                time.sleep(min(sleep_time, 60))
                continue

        print(f"❌ Failed to answer for {image_path.name} after {max_attempts_per_example} attempts. Last error: {last_exc}")
        return "❌ Failed to answer"


# ----------------- I/O helpers (persistence & files) -----------------

def load_vqa_annotations(path: Path):
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


def extract_index_from_answer(answer_text):
    import re
    if not answer_text:
        return None
    match = re.search(r'Index\s*:\s*(\d+)', str(answer_text))
    if match:
        return int(match.group(1))
    # fallback: try to find a small integer anywhere
    match2 = re.search(r'\b(\d)\b', str(answer_text))
    if match2:
        return int(match2.group(1))
    return None


def compute_accuracy(preds, gts):
    correct = 0
    total = len(preds)
    for pred_idx, gt_idx in zip(preds, gts):
        if pred_idx is None or gt_idx is None:
            continue
        if pred_idx == gt_idx:
            correct += 1
    return round(100.0 * correct / total, 2) if total > 0 else 0.0


# ----------------- Processing logic with resume support -----------------

def process_sector(sector_name, sector_cfg, key_list, prompt_mode="few", n_examples=None):
    print(f"\n==== Processing sector: {sector_name} (prompt_mode={prompt_mode}) ====")
    images_dir = sector_cfg["images"]
    annotation_file = sector_cfg["annotation"]

    vqa_data = load_vqa_annotations(annotation_file)
    vqa_data = vqa_data[:n_examples]

    if not vqa_data:
        print("⚠️ No VQA data for this sector.")
        return

    out_file = OUTPUT_ROOT / f"{sector_name}_vqa_gemini_{prompt_mode}.json"
    results_list, processed_ids = load_existing_results(out_file)

    client = RotatingGeminiVQA(key_list, MODEL)
    preds, gts = [], []

    examples_to_process = [item for item in vqa_data if item.get("image_id") not in processed_ids]
    print(f"Total examples: {len(vqa_data)}; already done: {len(processed_ids)}; to process: {len(examples_to_process)}")

    if not examples_to_process:
        print("✅ Nothing to process; computing metrics from existing results.")
        for r in results_list:
            preds.append(r.get("predicted_index"))
            gts.append(r.get("ground_truth_index"))
        metrics = {"Accuracy (%)": compute_accuracy(preds, gts), "n_examples": len(preds)}
        metrics_out = OUTPUT_ROOT / f"{sector_name}_vqa_metrics_gemini_{prompt_mode}.json"
        with open(metrics_out, "w", encoding="utf8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        return

    for item in tqdm(examples_to_process, desc=f"VQA ({sector_name})"):
        image_id = item.get("image_id")
        question = item.get("question")
        options = item.get("options")
        answer = item.get("answer")

        # ground truth index
        try:
            gt_index = options.index(answer)
        except Exception:
            gt_index = None

        # locate image
        img_path = None
        for ext in (".png", ".jpg", ".jpeg"):
            candidate = images_dir / f"{image_id}{ext}"
            if candidate.exists():
                img_path = candidate
                break
        if img_path is None:
            print(f"⚠️ Image not found for {image_id}")
            continue

        pred_text = client.ask_with_rotation(img_path, question, options, prompt_mode=prompt_mode, max_attempts_per_example=10)
        pred_idx = extract_index_from_answer(pred_text)

        preds.append(pred_idx)
        gts.append(gt_index)

        results_list.append({
            "image_id": image_id,
            "question": question,
            "options": options,
            "ground_truth_index": gt_index,
            "ground_truth_answer": answer,
            "predicted_index": pred_idx,
            "answer_text": pred_text,
        })

        save_results_atomic(out_file, results_list)

        # polite pause (tune as needed)
        time.sleep(5)

    metrics = {"Accuracy (%)": compute_accuracy(preds, gts), "n_examples": len(preds)}
    metrics_out = OUTPUT_ROOT / f"{sector_name}_vqa_metrics_gemini_{prompt_mode}.json"
    with open(metrics_out, "w", encoding="utf8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"✅ Finished sector {sector_name}. Results: {out_file}, Metrics: {metrics_out}")


def main():
    for prompt_mode in ["zero", "few"]:
        print(f"\n########### Starting run for prompt_mode={prompt_mode} ###########")
        for sector_name, cfg in SECTORS.items():
            try:
                process_sector(sector_name, cfg, KEY_LIST, prompt_mode=prompt_mode, n_examples=None)
            except Exception as e:
                print(f"❗ Error processing sector {sector_name} ({prompt_mode}-shot): {e}")


if __name__ == "__main__":
    main()

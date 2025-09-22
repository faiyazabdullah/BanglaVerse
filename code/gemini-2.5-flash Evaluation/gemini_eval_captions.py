#!/usr/bin/env python3
"""
gemini_rotate_and_metrics.py

- Rotates through provided Gemini API keys on failure (quota/429/etc).
- Resumes per-sector by loading existing caption JSON and continuing.
- Computes BLEU (sacrebleu), ROUGE-L (LCS-based), and BertScore F1.
- Persists progress after each image.
"""

import os
import json
import time
from pathlib import Path
from tqdm import tqdm
import numpy as np
import sacrebleu
from rouge_score import rouge_scorer  # retained only for import parity (not used for rouge-1)
import google.generativeai as genai
from bert_score import score as bertscore_score

# ---------------- CONFIG ----------------
KEY_LIST = [
    # "your-gemini-api-key-1",
    # "your-gemini-api-key-2",
    # Add more keys if you have them
]

MODEL = "gemini-2.5-flash"

# Allowed image extensions
IMG_EXTS = {".png", ".jpg", ".jpeg"}

# Prompt templates
PROMPT_ZERO_SHOT = (
    "You are an assistant that generates short, fluent captions in Bangla only. "
    "Look carefully at the given image and write exactly one meaningful sentence describing it. "
    "Do not use any English words, do not add extra explanations, labels, or quotes. "
    "Your entire output must be only the Bangla caption as plain text."
)

PROMPT_FEW_SHOT = (
    "You are an assistant that generates short, fluent captions in Bangla only.\n\n"
    "Examples:\n"
    "Image: ./dataset/history/images/history_002.png\n"
    "যুদ্ধজয়ে বীর বাঙালিদের সামনে এভাবেই বৈঠকে আত্মসমর্পণ করে পাকিস্তানি বাহিনী।\n\n"
    "Image: ./dataset/culture/images/culture_003.png\n"
    "পহেলা ফাল্গুনে রঙিন পোশাক পরে, ফুল দিয়ে সাজানো মানুষদের ভিড়।\n\n"
    "Image: ./dataset/sports/images/sports_001.png\n"
    "ছবির খেলোয়াড় আর কেউ নয়, বিশ্বসেরা অলরাউন্ডার সাকিব আল হাসান।\n\n"
    "Now, generate a caption for the following image. "
    "Write exactly one meaningful Bangla sentence. "
    "Do not use any English words, do not add extra explanations, labels, or quotes. "
    "Your entire output must be only the Bangla caption as plain text."
)

# Universal output folder for generated captions & metrics
OUTPUT_ROOT = Path(r"...\gemini_captions")  # Change this to your desired output directory
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# Sectors mapping: images dir and annotation file (gold)
# Ensure these paths are correct for your setup
SECTORS = {
    "culture": {
        "images": Path(r"...\data\culture\images"),
        "annotation": Path(r"...\data\culture\annotations\culture_captions.json")
    },
    "food": {
        "images": Path(r"...\data\food\images"),
        "annotation": Path(r"...\data\food\annotations\food_captions.json")
    },
    "history": {
        "images": Path(r"...\data\history\images"),
        "annotation": Path(r"...\data\history\annotations\history_captions.json")
    },
    "media_and_movies": {
        "images": Path(r"...\data\media_and_movies\images"),
        "annotation": Path(r"...\data\media_and_movies\annotations\media_and_movies_captions.json")
    },
    "national_achievements": {
        "images": Path(r"...\data\national_achievements\images"),
        "annotation": Path(r"...\data\national_achievements\annotations\national_achievements_captions.json")
    },
    "nature": {
        "images": Path(r"...\data\nature\images"),
        "annotation": Path(r"...\data\nature\annotations\nature_captions.json")
    },
    "personalities": {
        "images": Path(r"...\data\personalities\images"),
        "annotation": Path(r"...\data\personalities\annotations\personalities_captions.json")
    },
    "politics": {
        "images": Path(r"...\data\politics\images"),
        "annotation": Path(r"...\data\politics\annotations\politics_captions.json")
    },
    "sports": {
        "images": Path(r"...\data\sports\images"),
        "annotation": Path(r"...\data\sports\annotations\sports_captions.json")
    }
}

# ---------------- HELPERS ----------------
def load_gold_annotations(path: Path):
    if not path.exists():
        print(f"⚠️ Annotation file not found: {path}")
        return {}
    with open(path, "r", encoding="utf8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return data
    elif isinstance(data, list):
        out = {}
        for item in data:
            if "image_id" in item and ("caption" in item or "text" in item):
                out[item["image_id"]] = item.get("caption") or item.get("text")
        return out
    else:
        print("⚠️ Unknown annotation format; expected list or dict.")
        return {}

def list_images(folder: Path):
    if not folder.exists():
        print(f"⚠️ Image folder not found: {folder}")
        return []
    return sorted([p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS])

# LCS-based ROUGE-L (from your LLAMA script)
def lcs_length(pred, ref):
    """Compute the length of the Longest Common Subsequence (LCS)."""
    m, n = len(pred), len(ref)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred[i - 1] == ref[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]

def compute_rouge_l(pred, ref):
    """Calculate ROUGE-L F1 (character-level)."""
    lcs = lcs_length(pred, ref)
    precision = lcs / len(pred) if len(pred) > 0 else 0
    recall = lcs / len(ref) if len(ref) > 0 else 0
    if precision + recall == 0:
        return 0.0
    f_measure = (2 * precision * recall) / (precision + recall)
    return f_measure

# ---------------- METRICS ----------------
def compute_metrics_for_sector(preds_dict, gold_dict):
    preds, refs, skipped = [], [], []
    for imgid, pred in preds_dict.items():
        gold = gold_dict.get(imgid)
        if gold and pred and not pred.startswith("❌") and not pred.startswith("⚠️"):
            preds.append(pred)
            refs.append(gold)
        else:
            skipped.append(imgid)

    metrics = {"n_images_with_gold": len(preds), "n_skipped": len(skipped)}
    if len(preds) == 0:
        return metrics

    # BLEU
    try:
        bleu = sacrebleu.corpus_bleu(preds, [refs])
        metrics["BLEU"] = float(bleu.score)
    except Exception as e:
        print("BLEU calc failed:", e)
        metrics["BLEU"] = None

    # ROUGE-L via LCS
    try:
        rouge_l_scores = [compute_rouge_l(p, r) for p, r in zip(preds, refs)]
        metrics["ROUGE-L"] = float(np.mean(rouge_l_scores))
    except Exception as e:
        print("ROUGE-L calc failed:", e)
        metrics["ROUGE-L"] = None

    # BertScore (try rescaled; fallback to raw)
    try:
        P, R, F1 = bertscore_score(preds, refs, lang="bn", rescale_with_baseline=True)
        # F1 might be a tensor or numpy-like; use numpy to compute mean safely
        try:
            bert_f1 = float(np.mean(F1))
        except Exception:
            # fallback: try PyTorch tensor path
            try:
                import torch
                if isinstance(F1, torch.Tensor):
                    bert_f1 = float(F1.mean().item())
                else:
                    bert_f1 = float(np.mean(F1))
            except Exception:
                # last resort: cast to list and average
                bert_f1 = float(np.mean([float(x) for x in F1]))
        metrics["BertScore_F1"] = bert_f1
    except Exception as e:
        print(f"⚠️ BERTScore rescale failed: {e}. Falling back to raw BERTScore (rescale_with_baseline=False).")
        try:
            P, R, F1 = bertscore_score(preds, refs, lang="bn", rescale_with_baseline=False)
            try:
                bert_f1_raw = float(np.mean(F1))
            except Exception:
                import torch
                if isinstance(F1, torch.Tensor):
                    bert_f1_raw = float(F1.mean().item())
                else:
                    bert_f1_raw = float(np.mean(F1))
            # store both raw and a primary BertScore_F1 (so there's always a main scalar)
            metrics["BertScore_F1"] = bert_f1_raw
            metrics["BertScore_F1_raw"] = bert_f1_raw
        except Exception as e2:
            print(f"❗ BERTScore computation failed entirely: {e2}")
            metrics["BertScore_F1"] = None
            metrics["BertScore_F1_raw"] = None

    return metrics

# ----------------- Gemini captioner with rotating keys -----------------
class RotatingGeminiCaptioner:
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

    def generate_caption_once(self, image_path, prompt_text):
        """Single attempt using the currently configured key."""
        img = genai.upload_file(str(image_path))
        model = genai.GenerativeModel(self.model_name)
        resp = model.generate_content([prompt_text, img])
        # Safe parsing:
        if hasattr(resp, "text") and resp.text:
            return resp.text.strip()
        elif hasattr(resp, "candidates") and resp.candidates:
            cand_text = resp.candidates[0].content.parts[0].text
            return cand_text.strip()
        else:
            return "⚠️ Empty response"

    def generate_caption_with_rotation(self, image_path, prompt_text, max_attempts_per_image=10, backoff_base=1.0):
        """
        Try up to max_attempts_per_image (across keys). On quota/429 or other errors,
        rotate to the next key and retry the same image.
        """
        attempt = 0
        last_exception = None
        while attempt < max_attempts_per_image:
            attempt += 1
            try:
                caption = self.generate_caption_once(image_path, prompt_text)
                return caption
            except Exception as e:
                last_exception = e
                msg = str(e).lower()
                # immediate switch on quota/rate-limit
                if "429" in msg or "quota" in msg or "rate limit" in msg or "quota exceeded" in msg:
                    print(f"❗ Quota/rate-limit detected on key index {self.key_index}: {e}")
                    self._advance_key()
                    time.sleep(1.0)
                    continue
                # other transient errors -> rotate key and exponential backoff
                print(f"❗ API call failed on key index {self.key_index} (attempt {attempt}/{max_attempts_per_image}): {e}")
                self._advance_key()
                sleep_time = backoff_base * (2 ** (attempt - 1))
                print(f"⏳ backing off for {sleep_time:.1f}s before retrying this image")
                time.sleep(min(sleep_time, 60))
                continue

        print(f"❌ Failed to generate caption for {image_path.name} after {max_attempts_per_image} attempts. Last error: {last_exception}")
        return "❌ Failed to generate caption"

# ----------------- Persistence helpers -----------------
def load_existing_generated(out_file: Path):
    if out_file.exists():
        try:
            with open(out_file, "r", encoding="utf8") as f:
                data = json.load(f)
            generated_map = {item["image_id"]: item["caption"] for item in data if "image_id" in item}
            return data, generated_map
        except Exception as e:
            print(f"⚠️ Could not load existing generated file {out_file}: {e}")
            return [], {}
    else:
        return [], {}

def save_generated(out_file: Path, generated_list):
    tmp = out_file.with_suffix(out_file.suffix + ".tmp")
    with open(tmp, "w", encoding="utf8") as f:
        json.dump(generated_list, f, ensure_ascii=False, indent=2)
    tmp.replace(out_file)

# ----------------- Processing logic with resume support -----------------
def process_sector(sector_name, sector_cfg, captioner: RotatingGeminiCaptioner, prompt_mode="zero", n_examples=None):
    print(f"\n==== Processing sector: {sector_name} (prompt_mode={prompt_mode}) ====")
    images_dir = sector_cfg["images"]
    annotation_file = sector_cfg["annotation"]

    images = list_images(images_dir)
    if n_examples:
        images = images[:n_examples]

    if not images:
        print("⚠️ No images to process in this sector.")
        return

    prompt_text = PROMPT_ZERO_SHOT if prompt_mode == "zero" else PROMPT_FEW_SHOT

    out_file = OUTPUT_ROOT / f"{sector_name}_captions_gemini_{prompt_mode}.json"

    # Load existing generated to resume
    generated_list, generated_map = load_existing_generated(out_file)

    # Determine images to process (skip already done)
    images_to_process = [p for p in images if p.stem not in generated_map]
    print(f"Total images found: {len(images)}; already generated: {len(generated_map)}; to process: {len(images_to_process)}")

    if not images_to_process:
        print("✅ Nothing to process, computing metrics")
        gold = load_gold_annotations(annotation_file)
        metrics = compute_metrics_for_sector(generated_map, gold) if gold else {"error": "no_gold_annotations"}
        metrics_out = OUTPUT_ROOT / f"{sector_name}_metrics_gemini_{prompt_mode}.json"
        with open(metrics_out, "w", encoding="utf8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        return

    for img_path in tqdm(images_to_process, desc=f"Generating captions ({sector_name})"):
        image_id = img_path.stem
        caption = captioner.generate_caption_with_rotation(img_path, prompt_text, max_attempts_per_image=10)
        entry = {"image_id": image_id, "caption": caption}
        generated_list.append(entry)
        generated_map[image_id] = caption
        # persist after each image
        save_generated(out_file, generated_list)
        time.sleep(0.5)

    # compute & save metrics
    gold = load_gold_annotations(annotation_file)
    metrics = compute_metrics_for_sector(generated_map, gold) if gold else {"error": "no_gold_annotations"}
    metrics_out = OUTPUT_ROOT / f"{sector_name}_metrics_gemini_{prompt_mode}.json"
    with open(metrics_out, "w", encoding="utf8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"✅ Finished sector {sector_name}. Metrics written to {metrics_out}")

# ----------------- Main -----------------
def main():
    captioner = RotatingGeminiCaptioner(KEY_LIST, MODEL)
    for prompt_mode in ["zero", "few"]:
        for sector_name, cfg in SECTORS.items():
            try:
                process_sector(sector_name, cfg, captioner, prompt_mode=prompt_mode, n_examples=None)
            except Exception as e:
                print(f"❗ Error processing sector {sector_name} ({prompt_mode}-shot): {e}")

if __name__ == "__main__":
    main()

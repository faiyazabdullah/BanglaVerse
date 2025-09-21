#!/usr/bin/env python3
"""
gemini_csu_eval.py
Simplified: only Exact Match and BERTScore (with rescale fallback).
Run this from your normal conda/venv shell for best results.
"""

import os
import json
import time
from pathlib import Path
from tqdm import tqdm
import google.generativeai as genai
from bert_score import score as bertscore_score

API_KEY = "AIzaSyCpoEuW063lZTYYVWvWbcassLMA4SQzHd8"
MODEL = "gemini-2.5-flash"
IMG_EXTS = {".png", ".jpg", ".jpeg"}

OUTPUT_ROOT = Path(
    r"F:\Labib\Labib Folder\Labib\Research\BanglaVerse Experiments\BanglaVerse\gemini_csu"
)
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# --- SECTORS (unchanged) ---
SECTORS = {
    "culture": {
        "images": Path(
            r"F:\Labib\Labib Folder\Labib\Research\BanglaVerse Experiments\BanglaVerse\data\culture\images"
        ),
        "annotation": Path(
            r"F:\Labib\Labib Folder\Labib\Research\BanglaVerse Experiments\BanglaVerse\data\culture\annotations\culture_commonsense_reasoning.json"
        ),
    },
    "food": {
        "images": Path(
            r"F:\Labib\Labib Folder\Labib\Research\BanglaVerse Experiments\BanglaVerse\data\food\images"
        ),
        "annotation": Path(
            r"F:\Labib\Labib Folder\Labib\Research\BanglaVerse Experiments\BanglaVerse\data\food\annotations\food_commonsense_reasoning.json"
        ),
    },
    "history": {
        "images": Path(
            r"F:\Labib\Labib Folder\Labib\Research\BanglaVerse Experiments\BanglaVerse\data\history\images"
        ),
        "annotation": Path(
            r"F:\Labib\Labib Folder\Labib\Research\BanglaVerse Experiments\BanglaVerse\data\history\annotations\history_commonsense_reasoning.json"
        ),
    },
    "media_and_movies": {
        "images": Path(
            r"F:\Labib\Labib Folder\Labib\Research\BanglaVerse Experiments\BanglaVerse\data\media_and_movies\images"
        ),
        "annotation": Path(
            r"F:\Labib\Labib Folder\Labib\Research\BanglaVerse Experiments\BanglaVerse\data\media_and_movies\annotations\media_and_movies_commonsense_reasoning.json"
        ),
    },
    "national_achievements": {
        "images": Path(
            r"F:\Labib\Labib Folder\Labib\Research\BanglaVerse Experiments\BanglaVerse\data\national_achievements\images"
        ),
        "annotation": Path(
            r"F:\Labib\Labib Folder\Labib\Research\BanglaVerse Experiments\BanglaVerse\data\national_achievements\annotations\national_achievements_commonsense_reasoning.json"
        ),
    },
    "nature": {
        "images": Path(
            r"F:\Labib\Labib Folder\Labib\Research\BanglaVerse Experiments\BanglaVerse\data\nature\images"
        ),
        "annotation": Path(
            r"F:\Labib\Labib Folder\Labib\Research\BanglaVerse Experiments\BanglaVerse\data\nature\annotations\nature_commonsense_reasoning.json"
        ),
    },
    "personalities": {
        "images": Path(
            r"F:\Labib\Labib Folder\Labib\Research\BanglaVerse Experiments\BanglaVerse\data\personalities\images"
        ),
        "annotation": Path(
            r"F:\Labib\Labib Folder\Labib\Research\BanglaVerse Experiments\BanglaVerse\data\personalities\annotations\personalities_commonsense_reasoning.json"
        ),
    },
    "politics": {
        "images": Path(
            r"F:\Labib\Labib Folder\Labib\Research\BanglaVerse Experiments\BanglaVerse\data\politics\images"
        ),
        "annotation": Path(
            r"F:\Labib\Labib Folder\Labib\Research\BanglaVerse Experiments\BanglaVerse\data\politics\annotations\politics_commonsense_reasoning.json"
        ),
    },
    "sports": {
        "images": Path(
            r"F:\Labib\Labib Folder\Labib\Research\BanglaVerse Experiments\BanglaVerse\data\sports\images"
        ),
        "annotation": Path(
            r"F:\Labib\Labib Folder\Labib\Research\BanglaVerse Experiments\BanglaVerse\data\sports\annotations\sports_commonsense_reasoning.json"
        ),
    },
}

PROMPT_ZERO_SHOT = (
    "You are an assistant that answers Bangla cultural and commonsense reasoning questions about images. "
    "Look at the given {image_path} and the {question}. Provide a short, direct answer in Bangla."
)

PROMPT_FEW_SHOT = (
    "You are an assistant that answers Bangla cultural and commonsense reasoning questions about images. "
    "Look at the given {image_path} and the {question}. Provide a short, direct answer in Bangla.\n"
    "Examples:\n"
    "Image: ./dataset/history/images/history_002.png\n"
    "Question: \"ছবিতে কী হচ্ছে?\"\n"
    "Answer: \"বাংলাদেশের কাছে পাকিস্তান আত্মসমর্পণ করছে\"\n"
    "Image: ./dataset/culture/images/culture_003.png\n"
    "Question: \"ছবির এই দিনে মানুষ বাসন্তী রঙের পোশাক পরে, ফুল দিয়ে সাজে — দিনটি কী?\"\n"
    "Answer: \"পহেলা ফাল্গুন\"\n"
    "Image: ./dataset/sports/images/sports_001.png\n"
    "Question: \"ছবিতে সাকিব আল হাসান কিসের জন্য বিখ্যাত?\"\n"
    "Answer: \"অলরাউন্ডার হিসেবে\"\n"
    "Now, answer for the given image."
)


def normalize_text(text):
    import unicodedata, re
    if text is None:
        return ""
    t = unicodedata.normalize("NFKC", str(text))
    t = re.sub(r"\s+", " ", t).strip()
    return t


class GeminiCSU:
    def __init__(self, api_key, model):
        genai.configure(api_key=api_key)
        self.model = model

    def ask(self, image_path, question, prompt_mode="few", max_retries=3, sleep_between=1.0):
        if prompt_mode == "zero":
            prompt = PROMPT_ZERO_SHOT.format(image_path=str(image_path), question=question)
        else:
            prompt = PROMPT_FEW_SHOT + f"\nUser-input\n{str(image_path)}\n\"{question}\""
        for attempt in range(max_retries):
            try:
                img = genai.upload_file(str(image_path))
                model = genai.GenerativeModel(self.model)
                resp = model.generate_content([prompt, img])
                if hasattr(resp, "text") and resp.text:
                    return resp.text.strip()
                elif hasattr(resp, "candidates") and resp.candidates:
                    cand_text = resp.candidates[0].content.parts[0].text
                    return cand_text.strip()
                else:
                    return "⚠️ Empty response"
            except Exception as e:
                if "429" in str(e) or "quota" in str(e).lower():
                    print(f"❗ Gemini quota exceeded for {image_path.name}: {e}")
                    return "Answer not available due to Gemini API quota limits."
                print(f"❗ Gemini call failed for {image_path.name} (attempt {attempt+1}/{max_retries}): {e}")
                time.sleep(sleep_between * (attempt + 1))
        return "❌ Failed to answer"


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


# ---------- metrics (Exact Match + BERTScore) ----------
def compute_metrics(preds, refs):
    # Exact Match
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


def process_sector(sector_name, sector_cfg, prompt_mode="few", n_examples=10):
    print(f"\n==== Processing sector: {sector_name} (prompt_mode={prompt_mode}) ====")
    images_dir = sector_cfg["images"]
    annotation_file = sector_cfg["annotation"]

    csu_data = load_csu_annotations(annotation_file)
    csu_data = csu_data[:n_examples]  # Only process first n_examples examples

    if not csu_data:
        return

    csu_model = GeminiCSU(API_KEY, MODEL)
    results, preds, refs = [], [], []

    for item in tqdm(csu_data, desc=f"CSU ({sector_name})"):
        image_id = item.get("image_id")
        question = item.get("question")
        answer = item.get("answer")  # Reference answer

        img_path = images_dir / f"{image_id}.png"
        if not img_path.exists():
            img_path = images_dir / f"{image_id}.jpg"
        if not img_path.exists():
            img_path = images_dir / f"{image_id}.jpeg"
        if not img_path.exists():
            print(f"⚠️ Image not found for {image_id}")
            continue

        pred_answer = csu_model.ask(img_path, question, prompt_mode=prompt_mode)
        preds.append(pred_answer)
        refs.append(answer)
        results.append(
            {
                "image_id": image_id,
                "question": question,
                "ground_truth_answer": answer,
                "predicted_answer": pred_answer,
            }
        )
        time.sleep(20)  # Increased delay to avoid 429 errors

    # compute metrics
    metrics = compute_metrics(preds, refs)

    out_file = OUTPUT_ROOT / f"{sector_name}_csu_gemini_{prompt_mode}.json"
    with open(out_file, "w", encoding="utf8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    metrics_out = OUTPUT_ROOT / f"{sector_name}_csu_metrics_gemini_{prompt_mode}.json"
    with open(metrics_out, "w", encoding="utf8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)


def main():
    # Run only for "few-shot" mode
    for prompt_mode in ["few"]:  # Removed "zero" and kept only "few"
        for sector_name, cfg in SECTORS.items():
            try:
                process_sector(sector_name, cfg, prompt_mode=prompt_mode, n_examples=10)
            except Exception as e:
                print(f"❗ Error processing sector {sector_name} ({prompt_mode}-shot): {e}")


if __name__ == "__main__":
    main()
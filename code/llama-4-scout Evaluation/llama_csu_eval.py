#!/usr/bin/env python3
"""
llama_csu_eval.py
Simplified: only Exact Match and BERTScore (with rescale fallback).
Run this from your normal conda/venv shell for best results.
"""

import os
import json
import time
from pathlib import Path
from tqdm import tqdm
from groq import Groq
from bert_score import score as bertscore_score
import base64  # Import base64 for image encoding

# ---------------- CONFIG ----------------
API_KEYS = [
   "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" # Key 1
]
MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
IMG_EXTS = {".png", ".jpg", ".jpeg"}

OUTPUT_ROOT = Path(
    r"F:\Labib\Labib Folder\Labib\Research\BanglaVerse Experiments\BanglaVerse\llama_csu"
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


def encode_image_to_b64(path: Path) -> str:
    """Encode the image at the given path to a Base64 string."""
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        print(f"⚠️ Failed to encode image {path.name} to Base64: {e}")
        return None


class LlamaCSU:
    def __init__(self, api_keys, model):
        self.api_keys = api_keys
        self.model = model
        self.api_key_index = 0
        self.client = Groq(api_key=self.api_keys[self.api_key_index])

    def switch_api_key(self):
        """Switch to the next API key if the current one fails."""
        self.api_key_index += 1
        if self.api_key_index >= len(self.api_keys):
            raise RuntimeError("❌ All API keys exhausted.")
        self.client = Groq(api_key=self.api_keys[self.api_key_index])

    def ask(self, image_path, question, prompt_mode="few", max_retries=3, sleep_between=1.0):
        """Send a request to the Llama API with the given image and question."""
        # Encode the image to Base64
        b64_image = encode_image_to_b64(image_path)
        if not b64_image:
            return "❌ Failed to encode image"

        # Prepare the prompt
        prompt = (
            PROMPT_ZERO_SHOT.format(image_path=str(image_path), question=question)
            if prompt_mode == "zero"
            else PROMPT_FEW_SHOT + f"\nUser-input\n{str(image_path)}\n\"{question}\""
        )

        for attempt in range(max_retries):
            try:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}},
                        ],
                    }
                ]
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.2,
                    max_completion_tokens=128,
                    top_p=1,
                    stream=False,
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                print(f"❗ Llama API call failed for {image_path.name} (attempt {attempt+1}/{max_retries}): {e}")
                if "quota" in str(e).lower() or "429" in str(e):
                    print("⚠️ Switching API key due to quota limits.")
                    self.switch_api_key()
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

    csu_model = LlamaCSU(API_KEYS, MODEL)
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
        time.sleep(0.5)  # Reduced delay for Llama API

    # compute metrics
    metrics = compute_metrics(preds, refs)

    out_file = OUTPUT_ROOT / f"{sector_name}_csu_llama_{prompt_mode}.json"
    with open(out_file, "w", encoding="utf8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    metrics_out = OUTPUT_ROOT / f"{sector_name}_csu_metrics_llama_{prompt_mode}.json"
    with open(metrics_out, "w", encoding="utf8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)


def main():
    # Run only for few-shot mode and all sectors
    for prompt_mode in ["few"]:  # Only few-shot
        for sector_name in SECTORS.keys():  # All sectors
            try:
                process_sector(sector_name, SECTORS[sector_name], prompt_mode=prompt_mode, n_examples=10)
            except Exception as e:
                print(f"❗ Error processing sector {sector_name} ({prompt_mode}-shot): {e}")


if __name__ == "__main__":
    main()
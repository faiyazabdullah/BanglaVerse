import os
import json
import time
from pathlib import Path
from tqdm import tqdm
import numpy as np
import sacrebleu
from rouge_score import rouge_scorer
import google.generativeai as genai
from bert_score import score as bertscore_score  # Added BertScore

# --- 🔑 Direct API Key Setup ---
API_KEY = "AIzaSyBGQY6IZLVT_9A7psr8qhbdhd9v4TQaeRI"  # <-- replace with your Gemini API key
MODEL = "gemini-2.5-flash"   # change to gemini-2.5-flash

# Allowed image extensions
IMG_EXTS = {".png", ".jpg", ".jpeg"}

# Prompt templates
PROMPT_ZERO_SHOT = (
    "You are an assistant that generates short, fluent captions in Bangla. "
    "Look at the given image and describe it in one meaningful sentence in Bangla."
)

PROMPT_FEW_SHOT = (
    "You are an assistant that generates short, fluent captions in Bangla.\n\n"
    "Examples:\n"
    "Image: ./dataset/history/images/history_002.png\n"
    "Caption: \"যুদ্ধজয়ে বীর বাঙালিদের সামনে এভাবেই বৈঠকে আত্মসমর্পণ করে পাকিস্তানি বাহিনী।\"\n\n"
    "Image: ./dataset/culture/images/culture_003.png\n"
    "Caption: \"পহেলা ফাল্গুনে রঙিন পোশাক পরে, ফুল দিয়ে সাজানো মানুষদের ভিড়।\"\n\n"
    "Image: ./dataset/sports/images/sports_001.png\n"
    "Caption: \"ছবির খেলোয়াড় আর কেউ নয়, বিশ্বসেরা অলরাউন্ডার সাকিব আল হাসান।\"\n\n"
    "Now generate a caption for the following image in one meaningful Bangla sentence."
)

# Universal output folder for generated captions & metrics
OUTPUT_ROOT = Path(r"F:\Labib\Labib Folder\Labib\Research\BanglaVerse Experiments\BanglaVerse\gemini_captions")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# Sectors mapping: images dir and annotation file (gold)
SECTORS = {
    "culture": {
        "images": Path(r"F:\Labib\Labib Folder\Labib\Research\BanglaVerse Experiments\BanglaVerse\data\culture\images"),
        "annotation": Path(r"F:\Labib\Labib Folder\Labib\Research\BanglaVerse Experiments\BanglaVerse\data\culture\annotations\culture_captions.json")
    },
    "food": {
        "images": Path(r"F:\Labib\Labib Folder\Labib\Research\BanglaVerse Experiments\BanglaVerse\data\food\images"),
        "annotation": Path(r"F:\Labib\Labib Folder\Labib\Research\BanglaVerse Experiments\BanglaVerse\data\food\annotations\food_captions.json")
    },
    "history": {
        "images": Path(r"F:\Labib\Labib Folder\Labib\Research\BanglaVerse Experiments\BanglaVerse\data\history\images"),
        "annotation": Path(r"F:\Labib\Labib Folder\Labib\Research\BanglaVerse Experiments\BanglaVerse\data\history\annotations\history_captions.json")
    },
    "media_and_movies": {
        "images": Path(r"F:\Labib\Labib Folder\Labib\Research\BanglaVerse Experiments\BanglaVerse\data\media_and_movies\images"),
        "annotation": Path(r"F:\Labib\Labib Folder\Labib\Research\BanglaVerse Experiments\BanglaVerse\data\media_and_movies\annotations\media_and_movies_captions.json")
    },
    "national_achievements": {
        "images": Path(r"F:\Labib\Labib Folder\Labib\Research\BanglaVerse Experiments\BanglaVerse\data\national_achievements\images"),
        "annotation": Path(r"F:\Labib\Labib Folder\Labib\Research\BanglaVerse Experiments\BanglaVerse\data\national_achievements\annotations\national_achievements_captions.json")
    },
    "nature": {
        "images": Path(r"F:\Labib\Labib Folder\Labib\Research\BanglaVerse Experiments\BanglaVerse\data\nature\images"),
        "annotation": Path(r"F:\Labib\Labib Folder\Labib\Research\BanglaVerse Experiments\BanglaVerse\data\nature\annotations\nature_captions.json")
    },
    "personalities": {
        "images": Path(r"F:\Labib\Labib Folder\Labib\Research\BanglaVerse Experiments\BanglaVerse\data\personalities\images"),
        "annotation": Path(r"F:\Labib\Labib Folder\Labib\Research\BanglaVerse Experiments\BanglaVerse\data\personalities\annotations\personalities_captions.json")
    },
    "politics": {
        "images": Path(r"F:\Labib\Labib Folder\Labib\Research\BanglaVerse Experiments\BanglaVerse\data\politics\images"),
        "annotation": Path(r"F:\Labib\Labib Folder\Labib\Research\BanglaVerse Experiments\BanglaVerse\data\politics\annotations\politics_captions.json")
    },
    "sports": {
        "images": Path(r"F:\Labib\Labib Folder\Labib\Research\BanglaVerse Experiments\BanglaVerse\data\sports\images"),
        "annotation": Path(r"F:\Labib\Labib Folder\Labib\Research\BanglaVerse Experiments\BanglaVerse\data\sports\annotations\sports_captions.json")
    }
}

class GeminiCaptioner:
    def __init__(self, api_key, model):
        genai.configure(api_key=api_key)
        self.model = model

    def generate_caption(self, image_path, prompt_text, max_retries=3, sleep_between=1.0):
        """Generate caption for a single image using Gemini API."""
        quota_error_msg = "Caption not available due to Gemini API quota limits."
        for attempt in range(max_retries):
            try:
                img = genai.upload_file(str(image_path))
                model = genai.GenerativeModel(self.model)
                resp = model.generate_content([prompt_text, img])

                # --- Safe parsing of Gemini response ---
                if hasattr(resp, "text") and resp.text:
                    return resp.text.strip()
                elif hasattr(resp, "candidates") and resp.candidates:
                    cand_text = resp.candidates[0].content.parts[0].text
                    return cand_text.strip()
                else:
                    return "⚠️ Empty response"
            except Exception as e:
                # Detect quota error
                if "429" in str(e) or "quota" in str(e).lower():
                    print(f"❗ Gemini quota exceeded for {image_path.name}: {e}")
                    return quota_error_msg
                print(f"❗ Gemini call failed for {image_path.name} (attempt {attempt+1}/{max_retries}): {e}")
                time.sleep(sleep_between * (attempt+1))
        return "❌ Failed to generate caption"

# --- Helpers ---
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

def compute_metrics_for_sector(preds_dict, gold_dict):
    preds, refs, skipped = [], [], []
    for imgid, pred in preds_dict.items():
        gold = gold_dict.get(imgid)
        if gold and pred:
            preds.append(pred)
            refs.append(gold)
        else:
            skipped.append(imgid)

    metrics = {"n_images_with_gold": len(preds), "n_skipped": len(skipped)}

    if len(preds) == 0:
        return metrics

    try:
        bleu = sacrebleu.corpus_bleu(preds, [refs])
        metrics["BLEU"] = float(bleu.score)
    except Exception:
        metrics["BLEU"] = None

    try:
        scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
        r1_scores, rl_scores = [], []
        for p, r in zip(preds, refs):
            scr = scorer.score(r, p)
            r1_scores.append(scr['rouge1'].fmeasure)
            # rl_scores.append(scr['rougeL'].fmeasure)
        metrics["ROUGE-1"] = float(np.mean(r1_scores))
        # metrics["ROUGE-L"] = float(np.mean(rl_scores))
    except Exception:
        metrics["ROUGE-1"] = None
        # metrics["ROUGE-L"] = None

    try:
        P, R, F1 = bertscore_score(preds, refs, lang="bn", rescale_with_baseline=True)
        metrics["BertScore_F1"] = float(np.mean(F1))
    except Exception:
        metrics["BertScore_F1"] = None

    return metrics


def process_sector(sector_name, sector_cfg, prompt_mode="zero", n_examples=10):
    print(f"\n==== Processing sector: {sector_name} (prompt_mode={prompt_mode}) ====")
    images_dir = sector_cfg["images"]
    annotation_file = sector_cfg["annotation"]

    images = list_images(images_dir)
    images = images[:n_examples]  # Only process first 10 images

    if not images:
        return

    prompt_text = PROMPT_ZERO_SHOT if prompt_mode == "zero" else PROMPT_FEW_SHOT

    captioner = GeminiCaptioner(API_KEY, MODEL)
    generated, generated_map = [], {}
    for img_path in tqdm(images, desc=f"Generating captions ({sector_name})"):
        image_id = img_path.stem
        caption = captioner.generate_caption(img_path, prompt_text)
        entry = {"image_id": image_id, "caption": caption}
        generated_map[image_id] = caption
        generated.append(entry)
        time.sleep(0.5)

    out_file = OUTPUT_ROOT / f"{sector_name}_captions_gemini_{prompt_mode}.json"
    with open(out_file, "w", encoding="utf8") as f:
        json.dump(generated, f, ensure_ascii=False, indent=2)

    gold = load_gold_annotations(annotation_file)
    if not gold:
        metrics = {"error": "no_gold_annotations"}
    else:
        metrics = compute_metrics_for_sector(generated_map, gold)

    metrics_out = OUTPUT_ROOT / f"{sector_name}_metrics_gemini_{prompt_mode}.json"
    with open(metrics_out, "w", encoding="utf8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

def main():
    for prompt_mode in ["zero", "few"]:
        for sector_name, cfg in SECTORS.items():
            try:
                process_sector(sector_name, cfg, prompt_mode=prompt_mode, n_examples=10)
            except Exception as e:
                print(f"❗ Error processing sector {sector_name} ({prompt_mode}-shot): {e}")

if __name__ == "__main__":
    main()

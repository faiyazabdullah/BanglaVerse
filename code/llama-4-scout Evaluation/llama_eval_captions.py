# run_llama_all_sectors.py
import os
import json
import time
import base64
from pathlib import Path
from groq import Groq
import numpy as np
import sacrebleu
from bert_score import score as bertscore_score

# ---------------- CONFIG ----------------
API_KEYS = [
    # Add your Groq API keys here
]
MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

# Process all sectors and 10 examples per sector
N_EXAMPLES = None # set to None to process all images in each sector folder 

# Paths (adjust if needed)
OUTPUT_ROOT = Path(r"...\llama_captions")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

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

IMG_EXTS = {".png", ".jpg", ".jpeg"}

# ---------------- helpers ----------------
def list_images(folder: Path):
    if not folder.exists():
        print(f"⚠️ Image folder not found: {folder}")
        return []
    return sorted([p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS])

def encode_image_to_b64(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def load_gold_annotations(path: Path):
    if not path.exists():
        print(f"⚠️ Annotation file not found: {path}")
        return {}
    with open(path, "r", encoding="utf8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return data
    out = {}
    for item in data:
        if "image_id" in item and ("caption" in item or "text" in item):
            out[item["image_id"]] = item.get("caption") or item.get("text")
    return out

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
    """Calculate ROUGE-L score."""
    lcs = lcs_length(pred, ref)
    precision = lcs / len(pred) if len(pred) > 0 else 0
    recall = lcs / len(ref) if len(ref) > 0 else 0
    if precision + recall == 0:
        return 0.0
    f_measure = (2 * precision * recall) / (precision + recall)
    return f_measure

def compute_metrics_for_sector(preds_dict, gold_dict):
    preds, refs = [], []
    skipped = []
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
        rouge_l_scores = [compute_rouge_l(p, r) for p, r in zip(preds, refs)]
        metrics["ROUGE-L"] = float(np.mean(rouge_l_scores))
    except Exception:
        metrics["ROUGE-L"] = None

    try:
        P, R, F1 = bertscore_score(preds, refs, lang="bn", model_type="bert-base-multilingual-cased")
        metrics["BertScore_F1"] = float(F1.mean().item())
    except Exception as e:
        metrics["BertScore_F1"] = None
        print("BERTScore failed:", e)

    return metrics

# ---------------- main flow ----------------
def process_sector(sector_name, sector_cfg, prompt_mode, n_examples, api_key_index=0):
    print(f"\n==== Processing sector: {sector_name} (prompt_mode={prompt_mode}) ====")
    images_dir = sector_cfg["images"]
    annotation_file = sector_cfg["annotation"]

    images = list_images(images_dir)
    images = images[:n_examples]

    if not images:
        print(f"⚠️ No images found in {images_dir}")
        return

    prompt_text = PROMPT_ZERO_SHOT if prompt_mode == "zero" else PROMPT_FEW_SHOT

    generated = []
    generated_map = {}
    for img_path in images:
        image_id = img_path.stem
        b64 = encode_image_to_b64(img_path)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
                ]
            }
        ]

        while api_key_index < len(API_KEYS):
            try:
                client = Groq(api_key=API_KEYS[api_key_index])
                print(f"Sending {img_path.name} to model with API key {api_key_index + 1}...")
                completion = client.chat.completions.create(
                    model=MODEL,
                    messages=messages,
                    temperature=0.2,
                    max_completion_tokens=128,
                    top_p=1,
                    stream=False
                )
                caption = completion.choices[0].message.content.strip()
                break
            except Exception as e:
                print(f"❗ API key {api_key_index + 1} failed: {e}")
                api_key_index += 1
                if api_key_index >= len(API_KEYS):
                    print("❌ All API keys exhausted. Stopping.")
                    return

        generated.append({"image_id": image_id, "caption": caption})
        generated_map[image_id] = caption
        time.sleep(0.5)

    out_file = OUTPUT_ROOT / f"{sector_name}_captions_llama_{prompt_mode}.json"
    with open(out_file, "w", encoding="utf8") as f:
        json.dump(generated, f, ensure_ascii=False, indent=2)
    print("Saved captions to:", out_file)

    gold = load_gold_annotations(annotation_file)
    if not gold:
        metrics = {"error": "no_gold_annotations"}
    else:
        metrics = compute_metrics_for_sector(generated_map, gold)

    metrics_out = OUTPUT_ROOT / f"{sector_name}_metrics_llama_{prompt_mode}.json"
    with open(metrics_out, "w", encoding="utf8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print("Saved metrics to:", metrics_out)

def main():
    for prompt_mode in ["zero", "few"]:
        for sector_name, sector_cfg in SECTORS.items():
            try:
                process_sector(sector_name, sector_cfg, prompt_mode, n_examples=N_EXAMPLES)
            except Exception as e:
                print(f"❗ Error processing sector {sector_name} ({prompt_mode}-shot): {e}")

if __name__ == "__main__":
    main()

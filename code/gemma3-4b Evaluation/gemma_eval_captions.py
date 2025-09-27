#!/usr/bin/env python3
# run_ollama_all_sectors.py
import os
import json
import time
import re
from pathlib import Path
import numpy as np
import sacrebleu
from bert_score import score as bertscore_score
from ollama import Client

# ---------------- CONFIG ----------------
OLLAMA_URL = "http://localhost:11434"   # change if your Ollama server is elsewhere
MODEL = "gemma3:4b"                      # change to your model name
N_EXAMPLES = None                 # number of images to process per sector (set to None to process all)
OUTPUT_ROOT = Path(r"...\gemma_captions")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# Prompt modes
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

IMG_EXTS = {".png", ".jpg", ".jpeg"}

# Temperature and retry behavior
TEMPERATURE = 0.6
MAX_RETRIES = 3

# ---------------- helpers ----------------
def list_images(folder: Path):
    if not folder.exists():
        print(f"⚠️ Image folder not found: {folder}")
        return []
    return sorted([p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS])

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

# ---------------- main flow using Ollama ----------------
def extract_caption_from_ollama_response(resp):
    """
    Try multiple heuristics to extract a clean caption string from various possible Ollama responses,
    including metadata-string forms like: "model='...' response=' caption text' ...".
    """
    # If resp is a dict with typical keys, try them first
    if isinstance(resp, dict):
        # common keys
        for k in ("response", "output", "text", "result"):
            if k in resp and isinstance(resp[k], str) and resp[k].strip():
                return resp[k].strip()
        # fallback: try to find any string value
        for v in resp.values():
            if isinstance(v, str) and v.strip():
                return v.strip()
        # last resort: convert to JSON string
        try:
            return json.dumps(resp, ensure_ascii=False)
        except Exception:
            return str(resp)

    # If resp is already a string, handle it
    if isinstance(resp, str):
        s = resp.strip()
        # If the string contains "response='...'" style, extract inside quotes
        m = re.search(r"response\s*=\s*['\"](.*?)['\"]", s, flags=re.DOTALL)
        if m:
            return m.group(1).strip()
        # Sometimes key is resp='...' or output='...'
        m2 = re.search(r"(?:resp|output|result)\s*=\s*['\"](.*?)['\"]", s, flags=re.DOTALL)
        if m2:
            return m2.group(1).strip()
        # Else return original
        return s

    # Fallback
    return str(resp)

def clean_caption_string(raw_text):
    """Cleans prefixes, metadata, quotes, newlines; returns single-line caption."""
    if not raw_text:
        return ""

    # 1) Extract if the raw_text itself looks like a JSON blob that contains a nested string pattern
    raw = raw_text

    # Remove leading 'Caption:' or 'caption：' etc.
    raw = re.sub(r'^[Cc]aption[:：]\s*', '', raw).strip()

    # If we have "Image: name" echoed, remove it
    raw = re.sub(r'^Image[:：]\s*\S+\s*', '', raw).strip()

    # If metadata-style like "response=' ... '" exists anywhere, extract last occurrence
    m_all = re.findall(r"response\s*=\s*['\"](.*?)['\"]", raw, flags=re.DOTALL)
    if m_all:
        raw = m_all[-1].strip()

    # Remove wrapping quotes if present
    if (raw.startswith('"') and raw.endswith('"')) or (raw.startswith("'") and raw.endswith("'")):
        raw = raw[1:-1].strip()

    # Replace sequences of whitespace/newlines with a single space
    raw = re.sub(r'\s*\n+\s*', ' ', raw)
    raw = re.sub(r'\s+', ' ', raw).strip()

    # Strip any trailing metadata-looking fragments after the Bangla sentence (heuristic: if there is "model=" or "done=" etc)
    # Cut off at first occurrence of " model=" or " done=" or " context=" if present
    cut_idx = None
    meta_markers = [" model=", " done=", " done_reason=", " total_duration=", " load_duration=", " prompt_eval", " context="]
    for mk in meta_markers:
        i = raw.find(mk)
        if i != -1:
            if cut_idx is None or i < cut_idx:
                cut_idx = i
    if cut_idx is not None:
        raw = raw[:cut_idx].strip()
        # Trim trailing punctuation left from cut
        raw = raw.rstrip(',:; ')

    return raw

def looks_bangla_only(s):
    """
    Return True if the string does NOT contain Latin letters A-Za-z.
    Allow punctuation, numerals, and Bangla characters.
    """
    if not s:
        return False
    return re.search(r'[A-Za-z]', s) is None

def process_sector(sector_name, sector_cfg, prompt_mode, n_examples, ollama_client: Client):
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
        print(f"-> Generating caption for {img_path.name}")

        # Build prompt (we include image filename as context; model receives actual image in images=[])
        prompt_for_model = prompt_text + f"\n\nImage: {img_path.name}\n"

        caption = ""
        last_resp = None

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = ollama_client.generate(
                    model=MODEL,
                    prompt=prompt_for_model,
                    images=[str(img_path)],
                    options={
                        "num_predict": 128,
                        "temperature": TEMPERATURE,
                        # you may add "top_p": 0.95 or other options supported by your Ollama client
                    }
                )
                last_resp = resp
            except Exception as e:
                print(f"❗ Ollama request failed for {img_path.name} (attempt {attempt}): {e}")
                time.sleep(0.6)
                continue

            # extract potential caption using heuristics
            try:
                raw = extract_caption_from_ollama_response(resp)
            except Exception:
                raw = str(resp)

            cleaned = clean_caption_string(raw)

            # If empty or contains Latin letters, retry
            if not cleaned:
                print(f"⚠️ Empty caption from model on attempt {attempt}. Retrying...")
                time.sleep(0.4)
                continue

            if not looks_bangla_only(cleaned):
                print(f"⚠️ Caption contains Latin letters on attempt {attempt}. Retrying...")
                # sometimes the model returns metadata with Latin tokens; retry
                time.sleep(0.4)
                continue

            # Accept cleaned Bangla-only caption
            caption = cleaned
            break

        # If retries exhausted and no clean Bangla caption, attempt a final fallback cleaning of last_resp
        if not caption and last_resp is not None:
            try:
                raw_fb = extract_caption_from_ollama_response(last_resp)
            except Exception:
                raw_fb = str(last_resp)
            caption = clean_caption_string(raw_fb)
            # If still contains Latin letters, remove them (best-effort) — keep only characters not in [A-Za-z]
            if not looks_bangla_only(caption):
                # remove Latin letters but keep spaces/punctuations
                caption = re.sub(r'[A-Za-z]', '', caption).strip()
                # final cleanup of whitespace
                caption = re.sub(r'\s+', ' ', caption).strip()

        # Final safety: if caption is empty even after fallback, set to empty string (saved as empty)
        if not caption:
            caption = ""

        print(f"  -> Caption: {caption}")
        generated.append({"image_id": image_id, "caption": caption})
        generated_map[image_id] = caption

        time.sleep(0.5)

    # Save captions JSON (list of objects)
    out_file = OUTPUT_ROOT / f"{sector_name}_captions_ollama_{prompt_mode}.json"
    with open(out_file, "w", encoding="utf8") as f:
        json.dump(generated, f, ensure_ascii=False, indent=2)
    print("Saved captions to:", out_file)

    # Compute metrics if gold annotations exist
    gold = load_gold_annotations(annotation_file)
    if not gold:
        metrics = {"error": "no_gold_annotations"}
    else:
        metrics = compute_metrics_for_sector(generated_map, gold)

    metrics_out = OUTPUT_ROOT / f"{sector_name}_metrics_ollama_{prompt_mode}.json"
    with open(metrics_out, "w", encoding="utf8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print("Saved metrics to:", metrics_out)

def main():
    # create ollama client once
    try:
        client = Client(host=OLLAMA_URL)
        print("Connected to Ollama at", OLLAMA_URL)
    except Exception as e:
        print("❌ Could not connect to Ollama. Make sure the Ollama daemon is running and reachable.")
        print("Error:", e)
        return

    for prompt_mode in ["zero", "few"]:
        for sector_name, sector_cfg in SECTORS.items():
            try:
                process_sector(sector_name, sector_cfg, prompt_mode, n_examples=N_EXAMPLES, ollama_client=client)
            except Exception as e:
                print(f"❗ Error processing sector {sector_name} ({prompt_mode}-shot): {e}")

if __name__ == "__main__":
    main()

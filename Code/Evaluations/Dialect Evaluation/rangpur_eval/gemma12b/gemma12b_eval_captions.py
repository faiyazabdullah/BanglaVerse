#!/usr/bin/env python3
# run_ollama_all_sectors.py
import os
import json
import time
import re
import random
from pathlib import Path
import numpy as np
import sacrebleu
from bert_score import score as bertscore_score
from ollama import Client

# ---------------- CONFIG ----------------
OLLAMA_URL = "http://localhost:11434"   # change if your Ollama server is elsewhere
MODEL = "gemma3:12b"                      # change to your model name
N_EXAMPLES = None
OUTPUT_ROOT = Path(r"/mnt/wwn-0x500a0751e14d807e-part2/Labib/Labib Folder/Labib/Research/BanglaVerse_2/BanglaVerseV2 Experiment/Results/Dialects_result/Rangpur/eng_prompt/gemma12b/gemma_captions")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# Prompt modes
PROMPT_ZERO_SHOT = (
    "You are an assistant that generates short, fluent captions in Bangla Rangpur dialect only. "
    "Look carefully at the given image and write exactly one meaningful sentence describing it. "
    "Do not use any English words, do not add extra explanations, labels, or quotes. "
    "Your entire output must be only the Bangla Rangpur dialect caption as plain text."
)

PROMPT_FEW_SHOT_TEMPLATE = (
    "You are an assistant that generates short, fluent captions in Bangla Rangpur dialect only.\n\n"
    "Examples:\n{examples}\n"
    "Now, generate a caption for the following image. "
    "Write exactly one meaningful Bangla Rangpur dialect sentence. "
    "Do not use any English words, do not add extra explanations, labels, or quotes. "
    "Your entire output must be only the Bangla Rangpur dialect caption as plain text."
)

SECTORS = {
    "culture": {
        "images": Path(r"/mnt/wwn-0x500a0751e14d807e-part2/Labib/Labib Folder/Labib/Research/BanglaVerse_2/BanglaVerseV2 Experiment/data/pure_bn/culture/images"),
        "annotation": Path(r"/mnt/wwn-0x500a0751e14d807e-part2/Labib/Labib Folder/Labib/Research/BanglaVerse_2/BanglaVerseV2 Experiment/data/dialectual_data/Rangpur/Results/Captions/culture_captions_rangpur.json")
    },
    "food": {
        "images": Path(r"/mnt/wwn-0x500a0751e14d807e-part2/Labib/Labib Folder/Labib/Research/BanglaVerse_2/BanglaVerseV2 Experiment/data/pure_bn/food/images"),
        "annotation": Path(r"/mnt/wwn-0x500a0751e14d807e-part2/Labib/Labib Folder/Labib/Research/BanglaVerse_2/BanglaVerseV2 Experiment/data/dialectual_data/Rangpur/Results/Captions/food_captions_rangpur.json")
    },
    "history": {
        "images": Path(r"/mnt/wwn-0x500a0751e14d807e-part2/Labib/Labib Folder/Labib/Research/BanglaVerse_2/BanglaVerseV2 Experiment/data/pure_bn/history/images"),
        "annotation": Path(r"/mnt/wwn-0x500a0751e14d807e-part2/Labib/Labib Folder/Labib/Research/BanglaVerse_2/BanglaVerseV2 Experiment/data/dialectual_data/Rangpur/Results/Captions/history_captions_rangpur.json")
    },
    "media_and_movies": {
        "images": Path(r"/mnt/wwn-0x500a0751e14d807e-part2/Labib/Labib Folder/Labib/Research/BanglaVerse_2/BanglaVerseV2 Experiment/data/pure_bn/media_and_movies/images"),
        "annotation": Path(r"/mnt/wwn-0x500a0751e14d807e-part2/Labib/Labib Folder/Labib/Research/BanglaVerse_2/BanglaVerseV2 Experiment/data/dialectual_data/Rangpur/Results/Captions/media_and_movies_captions_rangpur.json")
    },
    "national_achievements": {
        "images": Path(r"/mnt/wwn-0x500a0751e14d807e-part2/Labib/Labib Folder/Labib/Research/BanglaVerse_2/BanglaVerseV2 Experiment/data/pure_bn/national_achievements/images"),
        "annotation": Path(r"/mnt/wwn-0x500a0751e14d807e-part2/Labib/Labib Folder/Labib/Research/BanglaVerse_2/BanglaVerseV2 Experiment/data/dialectual_data/Rangpur/Results/Captions/national_achievements_captions_rangpur.json")
    },
    "nature": {
        "images": Path(r"/mnt/wwn-0x500a0751e14d807e-part2/Labib/Labib Folder/Labib/Research/BanglaVerse_2/BanglaVerseV2 Experiment/data/pure_bn/nature/images"),
        "annotation": Path(r"/mnt/wwn-0x500a0751e14d807e-part2/Labib/Labib Folder/Labib/Research/BanglaVerse_2/BanglaVerseV2 Experiment/data/dialectual_data/Rangpur/Results/Captions/nature_captions_rangpur.json")
    },
    "personalities": {
        "images": Path(r"/mnt/wwn-0x500a0751e14d807e-part2/Labib/Labib Folder/Labib/Research/BanglaVerse_2/BanglaVerseV2 Experiment/data/pure_bn/personalities/images"),
        "annotation": Path(r"/mnt/wwn-0x500a0751e14d807e-part2/Labib/Labib Folder/Labib/Research/BanglaVerse_2/BanglaVerseV2 Experiment/data/dialectual_data/Rangpur/Results/Captions/personalities_captions_rangpur.json")
    },
    "politics": {
        "images": Path(r"/mnt/wwn-0x500a0751e14d807e-part2/Labib/Labib Folder/Labib/Research/BanglaVerse_2/BanglaVerseV2 Experiment/data/pure_bn/politics/images"),
        "annotation": Path(r"/mnt/wwn-0x500a0751e14d807e-part2/Labib/Labib Folder/Labib/Research/BanglaVerse_2/BanglaVerseV2 Experiment/data/dialectual_data/Rangpur/Results/Captions/politics_captions_rangpur.json")
    },
    "sports": {
        "images": Path(r"/mnt/wwn-0x500a0751e14d807e-part2/Labib/Labib Folder/Labib/Research/BanglaVerse_2/BanglaVerseV2 Experiment/data/pure_bn/sports/images"),
        "annotation": Path(r"/mnt/wwn-0x500a0751e14d807e-part2/Labib/Labib Folder/Labib/Research/BanglaVerse_2/BanglaVerseV2 Experiment/data/dialectual_data/Rangpur/Results/Captions/sports_captions_rangpur.json")
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

def get_random_caption_examples(gold_dict, images_dir, current_image_id, n_examples=3):
    """Get n random caption examples from the same sector, excluding the current image."""
    available = [(img_id, caption) for img_id, caption in gold_dict.items() if img_id != current_image_id]
    if len(available) < n_examples:
        n_examples = len(available)
    if n_examples == 0:
        return []
    
    selected = random.sample(available, n_examples)
    examples_with_images = []
    
    for img_id, caption in selected:
        # find image
        img_path = None
        for ext in IMG_EXTS:
            cand = images_dir / f"{img_id}{ext}"
            if cand.exists():
                img_path = cand
                break
        if img_path:
            examples_with_images.append({
                "image_id": img_id,
                "image_path": str(img_path),
                "caption": caption
            })
    
    return examples_with_images

def format_caption_examples(examples):
    """Format the caption examples for the few-shot prompt."""
    formatted = []
    for ex in examples:
        formatted.append(
            f"Image: {ex['image_path']}\n{ex['caption']}\n"
        )
    return "\n".join(formatted)

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
    # Handle ollama._types.GenerateResponse object
    if hasattr(resp, 'response'):
        return resp.response.strip() if resp.response else ""
    
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

    # Load gold annotations for few-shot examples
    gold = load_gold_annotations(annotation_file)

    generated = []
    generated_map = {}
    total_input_tokens = 0
    total_output_tokens = 0
    for img_path in images:
        image_id = img_path.stem
        print(f"-> Generating caption for {img_path.name}")

        # Build prompt and images list
        if prompt_mode == "zero":
            prompt_for_model = PROMPT_ZERO_SHOT + f"\n\nImage: {img_path.name}\n"
            images_list = [str(img_path)]
        else:
            # Get 3 random examples from the same domain (excluding current image)
            examples = get_random_caption_examples(gold, images_dir, image_id, n_examples=3)
            formatted_examples = format_caption_examples(examples)
            prompt_for_model = PROMPT_FEW_SHOT_TEMPLATE.format(examples=formatted_examples) + f"\n\nImage: {img_path.name}\n"
            # Include example images + current image
            images_list = [ex["image_path"] for ex in examples] + [str(img_path)]

        caption = ""
        last_resp = None
        input_tokens = 0
        output_tokens = 0

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = ollama_client.generate(
                    model=MODEL,
                    prompt=prompt_for_model,
                    images=images_list,
                    options={
                        "num_predict": 128,
                        "temperature": TEMPERATURE,
                        # you may add "top_p": 0.95 or other options supported by your Ollama client
                    }
                )
                last_resp = resp
                # Extract token counts if available
                if hasattr(resp, 'prompt_eval_count'):
                    input_tokens = resp.prompt_eval_count or 0
                    output_tokens = resp.eval_count or 0
                elif isinstance(resp, dict):
                    input_tokens = resp.get("prompt_eval_count", 0)
                    output_tokens = resp.get("eval_count", 0)
            except Exception as e:
                print(f"❗ Ollama request failed for {img_path.name} (attempt {attempt}): {e}")
                time.sleep(5)
                continue

            # extract potential caption using heuristics
            try:
                raw = extract_caption_from_ollama_response(resp)
            except Exception:
                raw = str(resp)
            
            # Debug: print raw response to diagnose extraction issues
            print(f"  [DEBUG] Raw response type: {type(resp)}")
            if isinstance(resp, dict):
                print(f"  [DEBUG] Response keys: {list(resp.keys())}")
                print(f"  [DEBUG] Response content preview: {str(resp)[:200]}")
            print(f"  [DEBUG] Extracted raw: {raw[:200] if raw else 'EMPTY'}")

            cleaned = clean_caption_string(raw)
            print(f"  [DEBUG] Cleaned caption: {cleaned[:200] if cleaned else 'EMPTY'}")

            # If empty or contains Latin letters, retry
            if not cleaned:
                print(f"⚠️ Empty caption from model on attempt {attempt}. Retrying...")
                time.sleep(5)
                continue

            if not looks_bangla_only(cleaned):
                print(f"⚠️ Caption contains Latin letters on attempt {attempt}. Retrying...")
                # sometimes the model returns metadata with Latin tokens; retry
                time.sleep(5)
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

        total_input_tokens += (input_tokens or 0)
        total_output_tokens += (output_tokens or 0)

        print(f"  -> Caption: {caption}")
        generated.append({"image_id": image_id, "caption": caption})
        generated_map[image_id] = caption

        time.sleep(5)

    # Save captions JSON (list of objects)
    out_file = OUTPUT_ROOT / f"{sector_name}_captions_ollama_{prompt_mode}.json"
    with open(out_file, "w", encoding="utf8") as f:
        json.dump(generated, f, ensure_ascii=False, indent=2)
    print("Saved captions to:", out_file)

    # Compute metrics if gold annotations exist
    if not gold:
        # Load gold if not already loaded (for zero-shot mode)
        gold = load_gold_annotations(annotation_file)
    
    if not gold:
        metrics = {"error": "no_gold_annotations"}
    else:
        metrics = compute_metrics_for_sector(generated_map, gold)
    
    # Add token information to metrics
    metrics["total_input_tokens"] = total_input_tokens
    metrics["total_output_tokens"] = total_output_tokens
    metrics["total_tokens"] = total_input_tokens + total_output_tokens

    metrics_out = OUTPUT_ROOT / f"{sector_name}_metrics_ollama_{prompt_mode}.json"
    with open(metrics_out, "w", encoding="utf8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print("Saved metrics to:", metrics_out)
    print(f"   Total tokens: {total_input_tokens + total_output_tokens} (input: {total_input_tokens}, output: {total_output_tokens})")

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
#!/usr/bin/env python3
"""
gemma_vqa_eval.py

VQA evaluation pipeline using a local Ollama (Gemma) model.
- Sends image file paths to Ollama
- Cleans and normalizes model outputs
- Extracts reported index and reported answer text
- Reconciles index vs text conflicts with exact and fuzzy matching (difflib)
- Saves per-sector JSON outputs and metrics
"""

import os
import json
import time
import re
import unicodedata
import difflib
from pathlib import Path
from tqdm import tqdm

# Ollama client import
try:
    from ollama import Client
except Exception as e:
    raise RuntimeError("ollama client not available. Install the ollama client package.") from e

# ---------------- CONFIG ----------------
LLM_URL = "http://localhost:11434"
MODEL_NAME = "gemma3:4b"
LLM_NUM_CTX = 4096
LLM_SEED = 0

N_EXAMPLES = None # set to None to use all examples in the annotation file

OUTPUT_ROOT = Path(r"...\gemma_vqa")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

IMG_EXTS = {".png", ".jpg", ".jpeg"}

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

# ---------------- PROMPTS ----------------
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

# ---------------- Helpers ----------------
def list_images(folder: Path):
    if not folder.exists():
        return []
    return sorted([p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS])

def load_vqa_annotations(path: Path):
    if not path.exists():
        return []
    with open(path, "r", encoding="utf8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else []

def find_image_path(images_dir: Path, image_id: str):
    for ext in (".png", ".jpg", ".jpeg"):
        candidate = images_dir / f"{image_id}{ext}"
        if candidate.exists():
            return str(candidate)
    return None

def normalize_text(text):
    """Unicode NFKC, collapse whitespace, strip"""
    if text is None:
        return ""
    t = unicodedata.normalize("NFKC", str(text))
    t = re.sub(r"\s+", " ", t).strip()
    return t

def safe_parse_response(resp):
    """
    Extract a textual preview from various possible ollama response shapes.
    """
    if resp is None:
        return ""
    if isinstance(resp, dict):
        for key in ("response", "output", "text", "result"):
            if key in resp:
                val = resp[key]
                if isinstance(val, str) and val.strip():
                    return val.strip()
                if isinstance(val, dict):
                    for k2 in ("content", "message", "text"):
                        if k2 in val and isinstance(val[k2], str):
                            return val[k2].strip()
        # fallback: join string values or stringify
        string_parts = [v for v in resp.values() if isinstance(v, str)]
        if string_parts:
            return "\n".join(string_parts).strip()
        return str(resp)
    elif isinstance(resp, str):
        return resp.strip()
    else:
        return str(resp)

# ---------------- Extraction & Matching Helpers ----------------
def extract_index_from_answer(answer_text):
    """Return integer index if found (Index: N or leading 'N'), else None."""
    if not isinstance(answer_text, str):
        return None
    t = normalize_text(answer_text)
    match = re.search(r'Index\s*:\s*(\d+)', t, flags=re.IGNORECASE)
    if match:
        return int(match.group(1))
    match2 = re.search(r'^\s*(\d+)[\.\)\s-]*', t)
    if match2:
        return int(match2.group(1))
    return None

def extract_answer_text_from_response(answer_text):
    """
    Extract the quoted Answer text like Answer: "মারমা" or the substring after 'Answer:'.
    Returns normalized extracted string (may be empty).
    """
    if not isinstance(answer_text, str):
        return ""
    t = answer_text
    # quoted variant
    m = re.search(r'Answer\s*:\s*["“”\'`]{1}\s*(.+?)\s*["“”\'`]{1}', t, flags=re.IGNORECASE)
    if m:
        return normalize_text(m.group(1))
    # after Answer:
    m2 = re.search(r'Answer\s*:\s*(.+)', t, flags=re.IGNORECASE)
    if m2:
        return normalize_text(m2.group(1))
    return normalize_text(t)

def match_text_to_options(pred_text, options):
    """
    Try exact normalized match first, else try difflib close match.
    Returns (matched_index_or_None, matched_text_or_empty, method)
    """
    if not options:
        return None, "", "none"
    norm_opts = [normalize_text(o) for o in options]
    # direct exact (raw)
    for i, o in enumerate(options):
        if pred_text == o:
            return i, o, "exact"
    # normalized exact
    for i, no in enumerate(norm_opts):
        if pred_text == no:
            return i, options[i], "norm_exact"
    # difflib on normalized strings
    close = difflib.get_close_matches(pred_text, norm_opts, n=1, cutoff=0.6)
    if close:
        idx = norm_opts.index(close[0])
        return idx, options[idx], "difflib"
    return None, "", "none"

# ---------------- Ollama wrapper ----------------
class OllamaLLM:
    def __init__(self, host: str, model: str, num_ctx: int = 4096, seed: int = 0):
        self.host = host
        self.model = model
        self.num_ctx = num_ctx
        self.seed = seed
        self.client = Client(host=self.host)

    def generate(self, prompt: str, image_path: str = None, max_tokens: int = 256):
        """
        Calls Ollama. Note: do NOT set format='text' (invalid for some clients).
        Returns (raw_response_obj, cleaned_text_preview).
        """
        gen_args = {
            "model": self.model,
            "prompt": prompt,
            "options": {
                "seed": self.seed,
                "num_ctx": self.num_ctx,
                "num_predict": max_tokens
            }
        }
        if image_path:
            gen_args["images"] = [image_path]

        resp = self.client.generate(**gen_args)  # may raise; caller handles retries
        raw_preview = safe_parse_response(resp)

        # regex: extract response='...' or output='...' if present
        m = re.search(r"(?:response|output)='(.*?)'", raw_preview)
        if m:
            extracted = m.group(1)
        else:
            m2 = re.search(r'(?:response|output)="(.*?)"', raw_preview)
            extracted = m2.group(1) if m2 else raw_preview

        cleaned = normalize_text(extracted)
        return resp, cleaned

# ---------------- Core processing ----------------
def compute_accuracy(preds, gts):
    correct = 0
    total = len(preds)
    for p, g in zip(preds, gts):
        if p == g and p is not None and g is not None:
            correct += 1
    return round(100.0 * correct / total, 2) if total > 0 else 0.0

def process_sector(ollama_llm: OllamaLLM, sector_name, sector_cfg, prompt_mode="few", n_examples=None):
    print(f"\n==== Processing sector: {sector_name} (prompt_mode={prompt_mode}) ====")
    images_dir = sector_cfg["images"]
    annotation_file = sector_cfg["annotation"]

    vqa_data = load_vqa_annotations(annotation_file)
    vqa_data = vqa_data[:n_examples]

    if not vqa_data:
        print(f"⚠️ No VQA data found in {annotation_file}")
        return

    generated = []
    pred_indices, gt_indices = [], []

    prompt_template = PROMPT_FEW_SHOT if prompt_mode == "few" else PROMPT_ZERO_SHOT

    for item in tqdm(vqa_data, desc=f"VQA ({sector_name})"):
        image_id = item.get("image_id")
        question = item.get("question")
        options = item.get("options", [])
        answer = item.get("answer")  # ground truth answer string

        # ground-truth index (programming index)
        gt_index = None
        try:
            if answer in options:
                gt_index = options.index(answer)
            else:
                if isinstance(answer, int) and 0 <= answer < len(options):
                    gt_index = int(answer)
        except Exception:
            gt_index = None

        image_path = find_image_path(images_dir, image_id)
        if image_path is None:
            print(f"⚠️ Image not found for {image_id} in {images_dir}")
            continue

        full_prompt = prompt_template.format(image_path=image_path, question=question, options=options)

        # call LLM with retries
        resp_preview = ""
        raw_resp = None
        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            try:
                raw_resp, resp_preview = ollama_llm.generate(full_prompt, image_path=image_path, max_tokens=128)
                break
            except Exception as e:
                print(f"❗ Ollama generate failed (attempt {attempt}/{max_attempts}): {e}")
                time.sleep(0.5 * attempt)
                if attempt == max_attempts:
                    print("❌ Failed after retries, skipping this example.")
                    raw_resp, resp_preview = None, ""

        answer_text_raw = resp_preview or ""
        reported_index = extract_index_from_answer(answer_text_raw)
        reported_text_extracted = extract_answer_text_from_response(answer_text_raw)

        # Reconciliation logic
        predicted_index_final = reported_index
        predicted_text_final = None
        match_reason = None

        if isinstance(reported_index, int) and 0 <= reported_index < len(options):
            option_at_index = options[reported_index]
            if reported_text_extracted:
                if normalize_text(option_at_index) == normalize_text(reported_text_extracted):
                    predicted_text_final = option_at_index
                    match_reason = "index_matches_text"
                else:
                    match_idx, match_text, method = match_text_to_options(reported_text_extracted, options)
                    if match_idx is not None:
                        predicted_index_final = match_idx
                        predicted_text_final = match_text
                        match_reason = f"index_text_conflict_resolved_by_{method}"
                    else:
                        predicted_text_final = option_at_index
                        match_reason = "index_trusted_text_mismatch"
            else:
                predicted_text_final = option_at_index
                match_reason = "index_only"
        else:
            # No valid reported index; try to match the reported text
            if reported_text_extracted:
                match_idx, match_text, method = match_text_to_options(reported_text_extracted, options)
                if match_idx is not None:
                    predicted_index_final = match_idx
                    predicted_text_final = match_text
                    match_reason = f"text_only_resolved_by_{method}"
                else:
                    predicted_index_final = None
                    predicted_text_final = reported_text_extracted
                    match_reason = "text_only_no_match"
            else:
                predicted_index_final = None
                predicted_text_final = ""
                match_reason = "no_prediction"

        pred_indices.append(predicted_index_final)
        gt_indices.append(gt_index)

        generated.append({
            "image_id": image_id,
            "question": question,
            "options": options,
            "ground_truth_index": gt_index,
            "ground_truth_answer": answer,
            "reported_index": reported_index,
            "reported_answer_text": answer_text_raw,
            "reported_answer_extracted_text": reported_text_extracted,
            "predicted_index": predicted_index_final,
            "predicted_answer_text": predicted_text_final,
            "match_reason": match_reason,
            # FIX: convert raw_resp to a string or drop it
            #"raw_response": str(raw_resp)  # or use resp_preview if you want only text
        })


        time.sleep(0.3)

    accuracy = compute_accuracy(pred_indices, gt_indices)
    metrics = {"Accuracy (%)": accuracy, "n_examples": len(generated)}

    out_file = OUTPUT_ROOT / f"{sector_name}_vqa_gemma_{prompt_mode}.json"
    with open(out_file, "w", encoding="utf8") as f:
        json.dump(generated, f, ensure_ascii=False, indent=2)
    print("Saved results to:", out_file)

    metrics_out = OUTPUT_ROOT / f"{sector_name}_vqa_metrics_gemma_{prompt_mode}.json"
    with open(metrics_out, "w", encoding="utf8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print("Saved metrics to:", metrics_out)
    print(f"Accuracy: {accuracy}% over {len(generated)} examples")

def main():
    ollama_llm = OllamaLLM(host=LLM_URL, model=MODEL_NAME, num_ctx=LLM_NUM_CTX, seed=LLM_SEED)
    for prompt_mode in ["zero", "few"]:
        for sector_name, sector_cfg in SECTORS.items():
            try:
                process_sector(ollama_llm, sector_name, sector_cfg, prompt_mode=prompt_mode, n_examples=N_EXAMPLES)
            except Exception as e:
                print(f"❗ Error processing sector {sector_name} ({prompt_mode}-shot): {e}")

if __name__ == "__main__":
    main()

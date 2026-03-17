#!/usr/bin/env python3
"""
Urdu VQA evaluation pipeline using a local Ollama (Gemma) model.
Adapted from original Bengali VQA script.

- VQA evaluation pipeline using a local Ollama (Gemma) model.
- Added Chain-of-Thought (CoT) prompt support (prompt_mode="cot").
- Parses CoT outputs to extract reasoning steps, reasoning text and predicted answer text.
- Atomic save + resume support (loads previous outputs and resumes).
- Output JSON schema extended to include reasoning_text, reasoning_steps_en, predicted_answer_text,
- in the same spirit as the Gemini rotating-keys script.
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

N_EXAMPLES = None

OUTPUT_ROOT = Path(r"/mnt/wwn-0x500a0751e14d807e-part2/Labib/Labib Folder/Labib/Research/BanglaVerse Experiments/BanglaVerse_V2/Results/urdu/gemma_vqa")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

IMG_EXTS = {".png", ".jpg", ".jpeg"}

SECTORS = {
    "culture": {
        "images": Path(r"/mnt/wwn-0x500a0751e14d807e-part2/Labib/Labib Folder/Labib/Research/BanglaVerse Experiments/BanglaVerse_V2/data_others/data_urdu/culture/images"),
        "annotation": Path(r"/mnt/wwn-0x500a0751e14d807e-part2/Labib/Labib Folder/Labib/Research/BanglaVerse Experiments/BanglaVerse_V2/data_others/data_urdu/culture/annotations/culture_qa_pairs.json")
    },
    "food": {
        "images": Path(r"/mnt/wwn-0x500a0751e14d807e-part2/Labib/Labib Folder/Labib/Research/BanglaVerse Experiments/BanglaVerse_V2/data_others/data_urdu/food/images"),
        "annotation": Path(r"/mnt/wwn-0x500a0751e14d807e-part2/Labib/Labib Folder/Labib/Research/BanglaVerse Experiments/BanglaVerse_V2/data_others/data_urdu/food/annotations/food_qa_pairs.json")
    },
    "history": {
        "images": Path(r"/mnt/wwn-0x500a0751e14d807e-part2/Labib/Labib Folder/Labib/Research/BanglaVerse Experiments/BanglaVerse_V2/data_others/data_urdu/history/images"),
        "annotation": Path(r"/mnt/wwn-0x500a0751e14d807e-part2/Labib/Labib Folder/Labib/Research/BanglaVerse Experiments/BanglaVerse_V2/data_others/data_urdu/history/annotations/history_qa_pairs.json")
    },
    "media_and_movies": {
        "images": Path(r"/mnt/wwn-0x500a0751e14d807e-part2/Labib/Labib Folder/Labib/Research/BanglaVerse Experiments/BanglaVerse_V2/data_others/data_urdu/media_and_movies/images"),
        "annotation": Path(r"/mnt/wwn-0x500a0751e14d807e-part2/Labib/Labib Folder/Labib/Research/BanglaVerse Experiments/BanglaVerse_V2/data_others/data_urdu/media_and_movies/annotations/media_and_movies_qa_pairs.json")
    },
    "national_achievements": {
        "images": Path(r"/mnt/wwn-0x500a0751e14d807e-part2/Labib/Labib Folder/Labib/Research/BanglaVerse Experiments/BanglaVerse_V2/data_others/data_urdu/national_achievements/images"),
        "annotation": Path(r"/mnt/wwn-0x500a0751e14d807e-part2/Labib/Labib Folder/Labib/Research/BanglaVerse Experiments/BanglaVerse_V2/data_others/data_urdu/national_achievements/annotations/national_achievements_qa_pairs.json")
    },
    "nature": {
        "images": Path(r"/mnt/wwn-0x500a0751e14d807e-part2/Labib/Labib Folder/Labib/Research/BanglaVerse Experiments/BanglaVerse_V2/data_others/data_urdu/nature/images"),
        "annotation": Path(r"/mnt/wwn-0x500a0751e14d807e-part2/Labib/Labib Folder/Labib/Research/BanglaVerse Experiments/BanglaVerse_V2/data_others/data_urdu/nature/annotations/nature_qa_pairs.json")
    },
    "personalities": {
        "images": Path(r"/mnt/wwn-0x500a0751e14d807e-part2/Labib/Labib Folder/Labib/Research/BanglaVerse Experiments/BanglaVerse_V2/data_others/data_urdu/personalities/images"),
        "annotation": Path(r"/mnt/wwn-0x500a0751e14d807e-part2/Labib/Labib Folder/Labib/Research/BanglaVerse Experiments/BanglaVerse_V2/data_others/data_urdu/personalities/annotations/personalities_qa_pairs.json")
    },
    "politics": {
        "images": Path(r"/mnt/wwn-0x500a0751e14d807e-part2/Labib/Labib Folder/Labib/Research/BanglaVerse Experiments/BanglaVerse_V2/data_others/data_urdu/politics/images"),
        "annotation": Path(r"/mnt/wwn-0x500a0751e14d807e-part2/Labib/Labib Folder/Labib/Research/BanglaVerse Experiments/BanglaVerse_V2/data_others/data_urdu/politics/annotations/politics_qa_pairs.json")
    },
    "sports": {
        "images": Path(r"/mnt/wwn-0x500a0751e14d807e-part2/Labib/Labib Folder/Labib/Research/BanglaVerse Experiments/BanglaVerse_V2/data_others/data_urdu/sports/images"),
        "annotation": Path(r"/mnt/wwn-0x500a0751e14d807e-part2/Labib/Labib Folder/Labib/Research/BanglaVerse Experiments/BanglaVerse_V2/data_others/data_urdu/sports/annotations/sports_qa_pairs.json")
    }
}

# ---------------- PROMPTS ----------------
PROMPT_ZERO_SHOT = (
    "You are an AI assistant that answers visual multiple-choice questions in Urdu.\n"
    "Task:\n"
    "1. Look carefully at the given image: {image_path}\n"
    "2. Read the question: {question}\n"
    "3. Review the provided answer choices: {options}\n"
    "4. Select the **single most accurate answer**.\n\n"
    "Response Rules:\n"
    "- The index must be the programming list index (starting from 0).\n"
    "- Respond ONLY with the exact format below.\n"
    "- Use Urdu text for the answer option.\n"
    "- Do NOT add explanations, extra words, reasoning steps, or anything outside the specified format.\n"
    "- Follow this exact structure:\n\n"
    "Index: <option_index>, Answer: \"<option_text_in_Urdu>\""
)

PROMPT_FEW_SHOT = (
    "You are an AI assistant that answers visual multiple-choice questions in Urdu.\n"
    "Task:\n"
    "1. Look carefully at the given image: {image_path}\n"
    "2. Read the question: {question}\n"
    "3. Review the provided answer choices: {options}\n"
    "4. Select the **single most accurate answer**.\n\n"
    "Response Rules:\n"
    "- The index must be the programming list index (starting from 0).\n"
    "- Respond ONLY with the exact format below.\n"
    "- Use Urdu text for the answer option.\n"
    "- Do NOT add explanations, extra words, reasoning steps, or anything outside the specified format.\n"
    "- Follow this exact structure:\n"
    "Index: <option_index>, Answer: \"<option_text_in_Urdu>\"\n\n"
    "Examples:\n"
    "Image: ./dataset/culture/images/culture_003.png\n"
    "Question: \"تصویر میں کون سا تہوار منایا جا رہا ہے؟\"\n"
    "Options: [\"پہلا بیساکھ\", \"عید\", \"نیا سال\", \"نبن\"]\n"
    "Answer: Index: 0, Answer: \"پہلا بیساکھ\"\n\n"
    "Image: ./dataset/history/images/history_002.png\n"
    "Question: \"اس واقعے کے بعد بنگلہ دیش میں کیا تبدیلی آئی؟\"\n"
    "Options: [\"آئین کا اعلان\", \"آزادی کا حصول\", \"فوجی حکومت کی شروعات\", \"ریفرنڈم\"]\n"
    "Answer: Index: 1, Answer: \"آزادی کا حصول\"\n\n"
    "Image: ./dataset/sports/images/sports_001.png\n"
    "Question: \"یہ کرکٹ کھلاڑی کون ہے؟\"\n"
    "Options: [\"مشرفی بن مرتضیٰ\", \"شاکب الحسن\", \"مشفق الرحیم\", \"تمیم اقبال\"]\n"
    "Answer: Index: 1, Answer: \"شاکب الحسن\"\n\n"
    "Now, answer for the given image."
)

PROMPT_CHAIN_OF_THOUGHTS = (
    "You are an AI assistant that answers visual multiple-choice questions in Urdu.\n\n"
    "Task:\n"
    "1. Look carefully at the given image: {image_path}\n"
    "2. Read the question: {question}\n"
    "3. Review the provided answer choices: {options}\n"
    "4. Select the **single most accurate answer**.\n\n"
    "Response Rules:\n"
    "- The index must be the programming list index (starting from 0).\n"
    "- Use Urdu text for the answer option.\n"
    "- In Reasoning_En, write step-by-step reasoning in English — break down the solution logically:\n"
    "  Step 1: Describe key visual observations.\n"
    "  Step 2: Match observations to relevant answer choices.\n"
    "  Step 3: Eliminate incorrect choices with brief justification.\n"
    "  Step 4: Conclude why the final choice is correct.\n"
    "- Be clear, concise, and factual (avoid overly long explanations).\n"
    "- Follow this exact response format:\n\n"
    "Reasoning_En:\n"
    "Step 1: <your_observations>\n"
    "Step 2: <your_matching_logic>\n"
    "Step 3: <your_elimination_of_wrong_options>\n"
    "Step 4: <your_final_choice_reasoning>\n\n"
    "Final Answer: Index: <option_index>, Answer: \"<option_text_in_Urdu>\""
)

# ---------------- Urdu text validation ----------------
def looks_urdu_only(text):
    """Check if text contains primarily Urdu script."""
    if not text or not isinstance(text, str):
        return False
    
    # Remove common punctuation and numbers
    cleaned = re.sub(r'[0-9\s\.,\?!\"\'\(\)\[\]\{\};:\-_/\\]+', '', text)
    if not cleaned:
        return False
    
    # Check for Arabic/Urdu script (U+0600-U+06FF, U+0750-U+077F)
    urdu_chars = 0
    total_chars = len(cleaned)
    
    for char in cleaned:
        if '\u0600' <= char <= '\u06FF' or '\u0750' <= char <= '\u077F':
            urdu_chars += 1
    
    # Consider it Urdu if at least 70% of non-punctuation characters are in Urdu script
    return (urdu_chars / total_chars) >= 0.7 if total_chars > 0 else False

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
def extract_index_from_answer_simple(answer_text):
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
    Extract the quoted Answer text like Answer: "مارما" or the substring after 'Answer:'.
    Returns normalized extracted string (may be empty).
    """
    if not isinstance(answer_text, str):
        return ""
    t = answer_text
    # quoted variant
    m = re.search(r'Answer\s*:\s*["""\'`]{1}\s*(.+?)\s*["""\'`]{1}', t, flags=re.IGNORECASE)
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

# ---------------- CoT-aware parser (inspired by Code 2) ----------------
def extract_index_from_answer_cot(answer_text):
    """
    Parse model output which may contain Reasoning_En block and Final Answer.
    Returns (idx_or_None, reasoning_steps_list_or_None, reasoning_text_or_None, predicted_answer_text_or_None)
    """
    if not answer_text:
        return None, None, None, None

    text = str(answer_text).strip()

    # Try to capture Reasoning_En block
    reasoning_block = None
    m_reason = re.search(r'Reasoning[_ ]?En\s*[:\-]?\s*(.*?)(?:Final\s*Answer|Final\s*Answer:|Index\s*:|\Z)', text, flags=re.IGNORECASE | re.DOTALL)
    if m_reason:
        reasoning_block = m_reason.group(1).strip()

    reasoning_steps = None
    if reasoning_block:
        steps = []
        for i in range(1, 9):
            m_step = re.search(r'(?:^|\n)\s*Step\s*' + str(i) + r'\s*[:\-]?\s*(.*?)(?=(?:\n\s*Step\s*' + str(i+1) + r'\b)|\Z)', reasoning_block, flags=re.IGNORECASE | re.DOTALL)
            if m_step:
                step_text = re.sub(r'^\s*Step\s*\d+\s*[:\-]?\s*', '', m_step.group(1), flags=re.IGNORECASE).strip()
                if step_text:
                    steps.append(step_text)
        if steps:
            reasoning_steps = steps
        else:
            lines = [ln.strip() for ln in reasoning_block.splitlines() if ln.strip()]
            if lines:
                reasoning_steps = lines

    # Extract final index
    m_final_idx = re.search(r'Final\s*Answer\s*[:\-]?\s*(?:Index\s*[:\-]?\s*(\d+))', text, flags=re.IGNORECASE)
    if m_final_idx:
        idx = int(m_final_idx.group(1))
    else:
        m_idx_any = re.search(r'\bIndex\s*[:\-]?\s*(\d+)\b', text, flags=re.IGNORECASE)
        idx = int(m_idx_any.group(1)) if m_idx_any else None

    # Extract predicted answer text after Answer:
    predicted_answer_text = None
    m_ans = re.search(r'Answer\s*[:\-]?\s*[""]?([^""\n]+)[""]?', text, flags=re.IGNORECASE)
    if m_ans:
        predicted_answer_text = m_ans.group(1).strip()

    reasoning_text = reasoning_block if reasoning_block else None

    return idx, reasoning_steps, reasoning_text, predicted_answer_text

# ---------------- Ollama wrapper ----------------
class OllamaLLM:
    def __init__(self, host: str, model: str, num_ctx: int = 4096, seed: int = 0):
        self.host = host
        self.model = model
        self.num_ctx = num_ctx
        self.seed = seed
        self.client = Client(host=self.host)

    def generate(self, prompt: str, image_path: str = None, max_tokens: int = None):
        """
        Calls Ollama. Note: do NOT set format='text' (invalid for some clients).
        Returns (raw_response_obj, cleaned_text_preview).
        """
        options = {
            "seed": self.seed,
            "num_ctx": self.num_ctx,
        }
        
        # Only add num_predict if max_tokens is specified
        if max_tokens is not None:
            options["num_predict"] = max_tokens
        
        gen_args = {
            "model": self.model,
            "prompt": prompt,
            "options": options
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

# ----------------- Persistence helpers (resume + atomic save) -----------------
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

# ---------------- Core processing & metrics ----------------
def compute_accuracy(preds, gts):
    """
    Compute accuracy only on pairs where both pred and gt are not None.
    Returns (accuracy_pct, valid_count, correct_count)
    """
    correct = 0
    valid = 0
    for pred_idx, gt_idx in zip(preds, gts):
        if pred_idx is None or gt_idx is None:
            continue
        valid += 1
        if pred_idx == gt_idx:
            correct += 1
    acc = (100.0 * correct / valid) if valid > 0 else 0.0
    return round(acc, 2), valid, correct

def process_sector(ollama_llm: OllamaLLM, sector_name, sector_cfg, prompt_mode="few", n_examples=50):
    print(f"\n==== Processing sector: {sector_name} (prompt_mode={prompt_mode}) ====")
    images_dir = sector_cfg["images"]
    annotation_file = sector_cfg["annotation"]

    vqa_data = load_vqa_annotations(annotation_file)
    vqa_data = vqa_data[:n_examples]

    if not vqa_data:
        print(f"⚠️ No VQA data found in {annotation_file}")
        return

    out_file = OUTPUT_ROOT / f"{sector_name}_vqa_gemma_{prompt_mode}.json"
    results_list, processed_ids = load_existing_results(out_file)

    # collect preds/gts for incremental metrics
    preds, gts = [], []

    prompt_template = PROMPT_FEW_SHOT if prompt_mode == "few" else PROMPT_ZERO_SHOT
    if prompt_mode == "cot":
        prompt_template = PROMPT_CHAIN_OF_THOUGHTS

    examples_to_process = [item for item in vqa_data if item.get("image_id") not in processed_ids]
    print(f"Total examples: {len(vqa_data)}; already done: {len(processed_ids)}; to process: {len(examples_to_process)}")

    for item in tqdm(examples_to_process, desc=f"VQA ({sector_name})"):
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

        # build prompt; we pass options as JSON-like string to preserve unicode
        full_prompt = prompt_template.format(image_path=image_path, question=question, options=json.dumps(options, ensure_ascii=False))

        # call LLM with retries
        resp_preview = ""
        raw_resp = None
        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            try:
                raw_resp, resp_preview = ollama_llm.generate(full_prompt, image_path=image_path)
                break
            except Exception as e:
                print(f"❗ Ollama generate failed (attempt {attempt}/{max_attempts}): {e}")
                time.sleep(0.5 * attempt)
                if attempt == max_attempts:
                    print("❌ Failed after retries, skipping this example.")
                    raw_resp, resp_preview = None, ""

        answer_text_raw = resp_preview or ""
        # choose parsing depending on prompt_mode
        if prompt_mode == "cot":
            reported_index, reasoning_steps, reasoning_text, predicted_answer_text = extract_index_from_answer_cot(answer_text_raw)
        else:
            reported_index = extract_index_from_answer_simple(answer_text_raw)
            predicted_answer_text = extract_answer_text_from_response(answer_text_raw)
            reasoning_text, reasoning_steps = None, None

        # Reconciliation logic (similar to your previous logic but now stores reasoning fields)
        predicted_index_final = reported_index
        predicted_text_final = None
        match_reason = None

        if isinstance(reported_index, int) and 0 <= reported_index < len(options):
            option_at_index = options[reported_index]
            if predicted_answer_text:
                if normalize_text(option_at_index) == normalize_text(predicted_answer_text):
                    predicted_text_final = option_at_index
                    match_reason = "index_matches_text"
                else:
                    match_idx, match_text, method = match_text_to_options(predicted_answer_text, options)
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
            if predicted_answer_text:
                match_idx, match_text, method = match_text_to_options(predicted_answer_text, options)
                if match_idx is not None:
                    predicted_index_final = match_idx
                    predicted_text_final = match_text
                    match_reason = f"text_only_resolved_by_{method}"
                else:
                    predicted_index_final = None
                    predicted_text_final = predicted_answer_text
                    match_reason = "text_only_no_match"
            else:
                predicted_index_final = None
                predicted_text_final = ""
                match_reason = "no_prediction"

        # Compose standardized result item (keeps original fields and adds reasoning fields)
        result_item = {
            "image_id": image_id,
            "question": normalize_text(question),
            "options": [normalize_text(x) for x in options],
            "ground_truth_index": int(gt_index) if gt_index is not None else None,
            "ground_truth_answer": answer,
            "reported_index": int(reported_index) if isinstance(reported_index, int) else None,
            "reported_answer_text": answer_text_raw,
            "reported_answer_extracted_text": predicted_answer_text if predicted_answer_text else None,
            "predicted_index": int(predicted_index_final) if predicted_index_final is not None else None,
            "predicted_answer_text": predicted_text_final if predicted_text_final else None,
            "match_reason": match_reason,
            "reasoning_steps_en": reasoning_steps if reasoning_steps else None,
        }

        results_list.append(result_item)
        save_results_atomic(out_file, results_list)

        preds.append(result_item.get("predicted_index"))
        gts.append(result_item.get("ground_truth_index"))

        # polite pause (tune as needed)
        time.sleep(0.3)

    # final metrics
    acc, valid_count, correct = compute_accuracy(preds, gts)
    reasoning_count = sum(1 for r in results_list if r.get("reasoning_text"))
    metrics = {"Accuracy (%)": acc, "n_examples_total": len(preds), "n_valid_evaluated": valid_count, "n_correct": correct, "n_with_reasoning": reasoning_count}
    metrics_out = OUTPUT_ROOT / f"{sector_name}_vqa_metrics_gemma_{prompt_mode}.json"
    with open(metrics_out, "w", encoding="utf8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("Saved results to:", out_file)
    print("Saved metrics to:", metrics_out)
    print(f"Accuracy: {acc}% over {len(preds)} evaluated examples (valid_count={valid_count})")

def main():
    ollama_llm = OllamaLLM(host=LLM_URL, model=MODEL_NAME, num_ctx=LLM_NUM_CTX, seed=LLM_SEED)
    for prompt_mode in ["zero", "few", "cot"]:
        for sector_name, sector_cfg in SECTORS.items():
            try:
                process_sector(ollama_llm, sector_name, sector_cfg, prompt_mode=prompt_mode, n_examples=N_EXAMPLES)
            except Exception as e:
                print(f"❗ Error processing sector {sector_name} ({prompt_mode}-shot): {e}")

if __name__ == "__main__":
    main()

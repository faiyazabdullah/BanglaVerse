#!/usr/bin/env python3
"""
gemma_vqa_eval.py

VQA evaluation pipeline using a local Ollama (Gemma) model.
- Sends image file paths to Ollama
- Cleans and normalizes model outputs
- Extracts reported index and reported answer text
- Reconciles index vs text conflicts with exact and fuzzy matching (difflib)
- Saves per-sector JSON outputs and metrics
- Includes zero-shot, few-shot, and chain-of-thought (CoT) prompting modes
"""

import os
import json
import time
import re
import random
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

OUTPUT_ROOT = Path(r"/mnt/wwn-0x500a0751e14d807e-part2/Labib/Labib Folder/Labib/Research/BanglaVerse_2/BanglaVerseV2 Experiment/Results/urdu/eng_prompt/gemma4b/gemma_vqa")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

IMG_EXTS = {".png", ".jpg", ".jpeg"}

SECTORS = {
    "culture": {
        "images": Path(r"/mnt/wwn-0x500a0751e14d807e-part2/Labib/Labib Folder/Labib/Research/BanglaVerse_2/BanglaVerseV2 Experiment/data/other_lang/data_urdu/culture/images"),
        "annotation": Path(r"/mnt/wwn-0x500a0751e14d807e-part2/Labib/Labib Folder/Labib/Research/BanglaVerse_2/BanglaVerseV2 Experiment/data/other_lang/data_urdu/culture/annotations/culture_qa_pairs.json")
    },
    "food": {
        "images": Path(r"/mnt/wwn-0x500a0751e14d807e-part2/Labib/Labib Folder/Labib/Research/BanglaVerse_2/BanglaVerseV2 Experiment/data/other_lang/data_urdu/food/images"),
        "annotation": Path(r"/mnt/wwn-0x500a0751e14d807e-part2/Labib/Labib Folder/Labib/Research/BanglaVerse_2/BanglaVerseV2 Experiment/data/other_lang/data_urdu/food/annotations/food_qa_pairs.json")
    },
    "history": {
        "images": Path(r"/mnt/wwn-0x500a0751e14d807e-part2/Labib/Labib Folder/Labib/Research/BanglaVerse_2/BanglaVerseV2 Experiment/data/other_lang/data_urdu/history/images"),
        "annotation": Path(r"/mnt/wwn-0x500a0751e14d807e-part2/Labib/Labib Folder/Labib/Research/BanglaVerse_2/BanglaVerseV2 Experiment/data/other_lang/data_urdu/history/annotations/history_qa_pairs.json")
    },
    "media_and_movies": {
        "images": Path(r"/mnt/wwn-0x500a0751e14d807e-part2/Labib/Labib Folder/Labib/Research/BanglaVerse_2/BanglaVerseV2 Experiment/data/other_lang/data_urdu/media_and_movies/images"),
        "annotation": Path(r"/mnt/wwn-0x500a0751e14d807e-part2/Labib/Labib Folder/Labib/Research/BanglaVerse_2/BanglaVerseV2 Experiment/data/other_lang/data_urdu/media_and_movies/annotations/media_and_movies_qa_pairs.json")
    },
    "national_achievements": {
        "images": Path(r"/mnt/wwn-0x500a0751e14d807e-part2/Labib/Labib Folder/Labib/Research/BanglaVerse_2/BanglaVerseV2 Experiment/data/other_lang/data_urdu/national_achievements/images"),
        "annotation": Path(r"/mnt/wwn-0x500a0751e14d807e-part2/Labib/Labib Folder/Labib/Research/BanglaVerse_2/BanglaVerseV2 Experiment/data/other_lang/data_urdu/national_achievements/annotations/national_achievements_qa_pairs.json")
    },
    "nature": {
        "images": Path(r"/mnt/wwn-0x500a0751e14d807e-part2/Labib/Labib Folder/Labib/Research/BanglaVerse_2/BanglaVerseV2 Experiment/data/other_lang/data_urdu/nature/images"),
        "annotation": Path(r"/mnt/wwn-0x500a0751e14d807e-part2/Labib/Labib Folder/Labib/Research/BanglaVerse_2/BanglaVerseV2 Experiment/data/other_lang/data_urdu/nature/annotations/nature_qa_pairs.json")
    },
    "personalities": {
        "images": Path(r"/mnt/wwn-0x500a0751e14d807e-part2/Labib/Labib Folder/Labib/Research/BanglaVerse_2/BanglaVerseV2 Experiment/data/other_lang/data_urdu/personalities/images"),
        "annotation": Path(r"/mnt/wwn-0x500a0751e14d807e-part2/Labib/Labib Folder/Labib/Research/BanglaVerse_2/BanglaVerseV2 Experiment/data/other_lang/data_urdu/personalities/annotations/personalities_qa_pairs.json")
    },
    "politics": {
        "images": Path(r"/mnt/wwn-0x500a0751e14d807e-part2/Labib/Labib Folder/Labib/Research/BanglaVerse_2/BanglaVerseV2 Experiment/data/other_lang/data_urdu/politics/images"),
        "annotation": Path(r"/mnt/wwn-0x500a0751e14d807e-part2/Labib/Labib Folder/Labib/Research/BanglaVerse_2/BanglaVerseV2 Experiment/data/other_lang/data_urdu/politics/annotations/politics_qa_pairs.json")
    },
    "sports": {
        "images": Path(r"/mnt/wwn-0x500a0751e14d807e-part2/Labib/Labib Folder/Labib/Research/BanglaVerse_2/BanglaVerseV2 Experiment/data/other_lang/data_urdu/sports/images"),
        "annotation": Path(r"/mnt/wwn-0x500a0751e14d807e-part2/Labib/Labib Folder/Labib/Research/BanglaVerse_2/BanglaVerseV2 Experiment/data/other_lang/data_urdu/sports/annotations/sports_qa_pairs.json")
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
    "- Use Urdu text (Urdu script) for the answer option.\n"
    "- Do NOT add explanations, extra words, reasoning steps, or anything outside the specified format.\n"
    "- Follow this exact structure:\n\n"
    "Index: <option_index>, Answer: \"<option_text_in_Urdu>\""
)


PROMPT_FEW_SHOT_TEMPLATE = (
    "You are an AI assistant that answers visual multiple-choice questions in Urdu.\n"
    "Task:\n"
    "1. Look carefully at the given image: {image_path}\n"
    "2. Read the question: {question}\n"
    "3. Review the provided answer choices: {options}\n"
    "4. Select the **single most accurate answer**.\n\n"
    "Response Rules:\n"
    "- The index must be the programming list index (starting from 0).\n"
    "- Respond ONLY with the exact format below.\n"
    "- Use Urdu text (Urdu script) for the answer option.\n"
    "- Do NOT add explanations, extra words, reasoning steps, or anything outside the specified format.\n"
    "- Follow this exact structure:\n"
    "Index: <option_index>, Answer: \"<option_text_in_Urdu>\"\n\n"
    "Examples:\n{examples}\n"
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
    "- Use Urdu text (Urdu script) for the answer option.\n"
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

def get_random_vqa_examples(vqa_data, images_dir, current_image_id, n_examples=3):
    """Get n random VQA examples from the same sector, excluding the current image."""
    available = [item for item in vqa_data if item.get("image_id") != current_image_id]
    if len(available) < n_examples:
        n_examples = len(available)
    if n_examples == 0:
        return []
    
    selected = random.sample(available, n_examples)
    examples_with_images = []
    
    for item in selected:
        image_id = item.get("image_id")
        img_path = find_image_path(images_dir, image_id)
        if img_path:
            # Get ground truth index
            options = item.get("options", [])
            answer = item.get("answer")
            gt_index = None
            try:
                if answer in options:
                    gt_index = options.index(answer)
                elif isinstance(answer, int) and 0 <= answer < len(options):
                    gt_index = int(answer)
            except Exception:
                gt_index = None
            
            if gt_index is not None:
                examples_with_images.append({
                    "image_id": image_id,
                    "image_path": img_path,
                    "question": item.get("question"),
                    "options": options,
                    "answer_index": gt_index,
                    "answer_text": options[gt_index] if 0 <= gt_index < len(options) else answer
                })
    
    return examples_with_images

def format_vqa_examples(examples):
    """Format the VQA examples for the few-shot prompt."""
    formatted = []
    for ex in examples:
        formatted.append(
            f"Image: {ex['image_path']}\n"
            f"Question: \"{ex['question']}\"\n"
            f"Options: {json.dumps(ex['options'], ensure_ascii=False)}\n"
            f"Answer: Index: {ex['answer_index']}, Answer: \"{ex['answer_text']}\"\n"
        )
    return "\n".join(formatted)

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
    elif hasattr(resp, 'response'):
        return getattr(resp, 'response', '')
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
    m = re.search(r'Answer\s*:\s*[""'"'"'`]{1}\s*(.+?)\s*[""'"'"'`]{1}', t, flags=re.IGNORECASE)
    if m:
        return normalize_text(m.group(1))
    # after Answer:
    m2 = re.search(r'Answer\s*:\s*(.+)', t, flags=re.IGNORECASE)
    if m2:
        return normalize_text(m2.group(1))
    return normalize_text(t)

def extract_reasoning_and_final_answer(answer_text):
    """
    Extract reasoning steps and final answer from CoT response.
    Returns tuple: (reasoning_steps, reasoning_text, predicted_index, predicted_answer_text)
    """
    if not answer_text:
        return None, None, None, None

    text = str(answer_text).strip()

    # Helper: clean a step label like "Step 1:" => returns remainder
    def _clean_step(s):
        return re.sub(r'^\s*Step\s*\d+\s*[:\-]?\s*', '', s, flags=re.IGNORECASE).strip()

    # 1) Try to find Reasoning_En block then Final Answer (CoT)
    reasoning_block = None
    m_reason = re.search(r'Reasoning[_ ]?En\s*[:\-]?\s*(.*?)(?:Final\s*Answer|Index\s*:|\Z)', text, flags=re.IGNORECASE | re.DOTALL)
    if m_reason:
        reasoning_block = m_reason.group(1).strip()

    reasoning_steps = None
    if reasoning_block:
        # attempt to extract up to 8 Step N items in order
        steps = []
        for i in range(1, 9):
            m_step = re.search(r'(?:^|\n)\s*Step\s*' + str(i) + r'\s*[:\-]?\s*(.*?)(?=(?:\n\s*Step\s*' + str(i+1) + r'\b)|\Z)', reasoning_block, flags=re.IGNORECASE | re.DOTALL)
            if m_step:
                step_text = _clean_step(m_step.group(1))
                if step_text:
                    steps.append(step_text)
        if steps:
            reasoning_steps = steps
        else:
            lines = [ln.strip() for ln in reasoning_block.splitlines() if ln.strip()]
            if lines:
                reasoning_steps = lines

    # 2) Extract index via "Final Answer: Index: X" or "Final Answer - Index X"
    m_final_idx = re.search(r'Final\s*Answer\s*[:\-]?\s*(?:Index\s*[:\-]?\s*(\d+))', text, flags=re.IGNORECASE)
    if m_final_idx:
        idx = int(m_final_idx.group(1))
    else:
        # 3) Try "Index: X" anywhere else
        m_idx_any = re.search(r'\bIndex\s*[:\-]?\s*(\d+)\b', text, flags=re.IGNORECASE)
        idx = int(m_idx_any.group(1)) if m_idx_any else None

    # 4) Extract predicted answer text if present: look for Answer: "..." after Index or Final Answer
    predicted_answer_text = None
    m_ans = re.search(r'Answer\s*[:\-]?\s*[""'"'"']?([^"'"'"'\n]+)[""'"'"']?', text, flags=re.IGNORECASE)
    if m_ans:
        predicted_answer_text = m_ans.group(1).strip()

    # 5) Also attempt to extract reasoning_text more generically if not captured above
    reasoning_text = None
    if reasoning_block:
        reasoning_text = reasoning_block
    else:
        m_reason2 = re.search(r'Reasoning\s*[:\-]\s*(.*?)(?:Final\s*Answer|Index\s*:|\Z)', text, flags=re.IGNORECASE | re.DOTALL)
        reasoning_text = m_reason2.group(1).strip() if m_reason2 else None
        if reasoning_text:
            lines = [ln.strip() for ln in reasoning_text.splitlines() if ln.strip()]
            reasoning_steps = lines if lines else reasoning_steps

    return reasoning_steps, reasoning_text, idx, predicted_answer_text

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

    def generate(self, prompt: str, image_path: str = None, images: list = None, max_tokens: int = None):
        """
        Calls Ollama. Note: do NOT set format='text' (invalid for some clients).
        Returns (raw_response_obj, cleaned_text_preview, input_tokens, output_tokens).
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
        # Support both single image and multiple images
        if images:
            gen_args["images"] = images
        elif image_path:
            gen_args["images"] = [image_path]

        resp = self.client.generate(**gen_args)  # may raise; caller handles retries
        raw_preview = safe_parse_response(resp)
        
        # Extract token counts
        input_tokens = 0
        output_tokens = 0
        if isinstance(resp, dict):
            input_tokens = resp.get("prompt_eval_count", 0)
            output_tokens = resp.get("eval_count", 0)
        elif hasattr(resp, 'prompt_eval_count'):
            input_tokens = getattr(resp, 'prompt_eval_count', None) or 0
            output_tokens = getattr(resp, 'eval_count', None) or 0

        # regex: extract response='...' or output='...' if present
        m = re.search(r"(?:response|output)='(.*?)'", raw_preview)
        if m:
            extracted = m.group(1)
        else:
            m2 = re.search(r'(?:response|output)="(.*?)"', raw_preview)
            extracted = m2.group(1) if m2 else raw_preview

        cleaned = normalize_text(extracted)
        return resp, cleaned, input_tokens, output_tokens

# ---------------- Core processing ----------------
def compute_accuracy(preds, gts):
    correct = 0
    valid = 0
    for p, g in zip(preds, gts):
        if p is not None and g is not None:
            valid += 1
            if p == g:
                correct += 1
    return round(100.0 * correct / valid, 2) if valid > 0 else 0.0, valid, correct

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

def process_sector(ollama_llm: OllamaLLM, sector_name, sector_cfg, prompt_mode="few", n_examples=None):
    print(f"\n==== Processing sector: {sector_name} (prompt_mode={prompt_mode}) ====")
    images_dir = sector_cfg["images"]
    annotation_file = sector_cfg["annotation"]

    vqa_data = load_vqa_annotations(annotation_file)
    vqa_data = vqa_data[:n_examples]

    if not vqa_data:
        print(f"⚠️ No VQA data found in {annotation_file}")
        return

    out_file = OUTPUT_ROOT / f"{sector_name}_vqa_gemma12b_{prompt_mode}.json"
    results_list, processed_ids = load_existing_results(out_file)

    total_input_tokens = 0
    total_output_tokens = 0

    # Choose prompt template
    if prompt_mode == "zero":
        prompt_template = PROMPT_ZERO_SHOT
    elif prompt_mode == "cot":
        prompt_template = PROMPT_CHAIN_OF_THOUGHTS
    else:
        prompt_template = PROMPT_FEW_SHOT_TEMPLATE

    examples_to_process = [item for item in vqa_data if item.get("image_id") not in processed_ids]
    print(f"Total examples: {len(vqa_data)}; already done: {len(processed_ids)}; to process: {len(examples_to_process)}")

    if not examples_to_process:
        print("✅ Nothing to process; computing metrics from existing results.")
        preds = [r.get("predicted_index") for r in results_list]
        gts = [r.get("ground_truth_index") for r in results_list]
        acc, valid_count, correct = compute_accuracy(preds, gts)
        reasoning_count = sum(1 for r in results_list if r.get("reasoning_text"))
        metrics = {
            "Accuracy (%)": acc, 
            "n_examples_total": len(preds), 
            "n_valid_evaluated": valid_count, 
            "n_correct": correct,
            "n_with_reasoning": reasoning_count
        }
        metrics_out = OUTPUT_ROOT / f"{sector_name}_vqa_metrics_gemma12b_{prompt_mode}.json"
        with open(metrics_out, "w", encoding="utf8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f"Metrics computed from existing results: Accuracy = {acc}%")
        return

    pred_indices, gt_indices = [], []

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

        # Build prompt and images list
        if prompt_mode == "few":
            # Get 3 random examples from the same domain (excluding current image)
            examples = get_random_vqa_examples(vqa_data, images_dir, image_id, n_examples=3)
            formatted_examples = format_vqa_examples(examples)
            full_prompt = prompt_template.format(
                examples=formatted_examples,
                image_path=image_path,
                question=question,
                options=json.dumps(options, ensure_ascii=False)
            )
            # Include example images + current image
            images_list = [ex["image_path"] for ex in examples] + [image_path]
        else:
            full_prompt = prompt_template.format(
                image_path=image_path,
                question=question,
                options=json.dumps(options, ensure_ascii=False)
            )
            images_list = [image_path]

        # call LLM with retries
        resp_preview = ""
        raw_resp = None
        input_tokens = 0
        output_tokens = 0
        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            try:
                raw_resp, resp_preview, input_tokens, output_tokens = ollama_llm.generate(
                    full_prompt, images=images_list
                )
                break
            except Exception as e:
                print(f"❗ Ollama generate failed (attempt {attempt}/{max_attempts}): {e}")
                time.sleep(0.5 * attempt)
                if attempt == max_attempts:
                    print("❌ Failed after retries, skipping this example.")
                    raw_resp, resp_preview, input_tokens, output_tokens = None, "", 0, 0
        
        total_input_tokens += (input_tokens or 0)
        total_output_tokens += (output_tokens or 0)

        answer_text_raw = resp_preview or ""
        
        # Different extraction logic for CoT vs other modes
        if prompt_mode == "cot":
            reasoning_steps, reasoning_text, predicted_index, predicted_answer_text = extract_reasoning_and_final_answer(answer_text_raw)
            
            # If CoT extraction failed, fall back to standard extraction
            if predicted_index is None:
                predicted_index = extract_index_from_answer(answer_text_raw)
                predicted_answer_text = extract_answer_text_from_response(answer_text_raw)
                reasoning_steps = None
                reasoning_text = None
        else:
            predicted_index = extract_index_from_answer(answer_text_raw)
            predicted_answer_text = extract_answer_text_from_response(answer_text_raw)
            reasoning_steps = None
            reasoning_text = None

        # Text-to-option matching for cases where index is missing but text is present
        if predicted_index is None and predicted_answer_text:
            try:
                norm_pred = normalize_text(predicted_answer_text).lower()
                mapped = None
                for i, opt in enumerate(options):
                    if normalize_text(opt).lower() == norm_pred:
                        mapped = i
                        break
                predicted_index = mapped
            except Exception:
                predicted_index = None

        # Create result item matching Code 2's structure
        result_item = {
            "image_id": str(image_id),
            "question": normalize_text(question),
            "options": [normalize_text(x) for x in options],
            "ground_truth_index": int(gt_index) if gt_index is not None else None,
            "ground_truth_answer": normalize_text(answer) if answer is not None else None,
            "predicted_index": int(predicted_index) if predicted_index is not None else None,
            #"answer_text": normalize_text(answer_text_raw),
            #"predicted_answer_text": normalize_text(predicted_answer_text) if predicted_answer_text else None,
            #"reasoning_text": reasoning_text if reasoning_text else None,
            "reasoning_steps_en": reasoning_steps if reasoning_steps else None,
        }

        results_list.append(result_item)
        save_results_atomic(out_file, results_list)

        pred_indices.append(predicted_index)
        gt_indices.append(gt_index)

        time.sleep(0.3)

    # Compute final metrics
    acc, valid_count, correct = compute_accuracy(pred_indices, gt_indices)
    reasoning_count = sum(1 for r in results_list if r.get("reasoning_text"))
    metrics = {
        "Accuracy (%)": acc, 
        "n_examples_total": len(pred_indices), 
        "n_valid_evaluated": valid_count, 
        "n_correct": correct,
        "n_with_reasoning": reasoning_count,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_tokens": total_input_tokens + total_output_tokens
    }

    metrics_out = OUTPUT_ROOT / f"{sector_name}_vqa_metrics_gemma12b_{prompt_mode}.json"
    with open(metrics_out, "w", encoding="utf8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Finished sector {sector_name}")
    print(f"Results: {out_file}, Metrics: {metrics_out}")
    print(f"Accuracy: {acc}% over {valid_count} valid examples (total: {len(pred_indices)})")
    print(f"Total tokens: {total_input_tokens + total_output_tokens} (input: {total_input_tokens}, output: {total_output_tokens})")
    if reasoning_count > 0:
        print(f"Examples with reasoning: {reasoning_count}")

def main():
    ollama_llm = OllamaLLM(host=LLM_URL, model=MODEL_NAME, num_ctx=LLM_NUM_CTX, seed=LLM_SEED)
    
    # Run three prompt modes: zero-shot, few-shot, and chain-of-thought
    for prompt_mode in ["zero","few","cot"]:
        print(f"\n########### Starting run for prompt_mode={prompt_mode} ###########")
        for sector_name, sector_cfg in SECTORS.items():
            try:
                process_sector(ollama_llm, sector_name, sector_cfg, prompt_mode=prompt_mode, n_examples=N_EXAMPLES)
            except Exception as e:
                print(f"❗ Error processing sector {sector_name} ({prompt_mode}-shot): {e}")

if __name__ == "__main__":
    main()
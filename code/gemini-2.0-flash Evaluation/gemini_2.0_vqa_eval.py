#!/usr/bin/env python3
"""
gemini_vqa_rotate_keys_run_both_modes_with_cot.py

- Rotates through multiple Gemini API keys on failure (quota/429/etc).
- Retries the same example after switching keys (doesn't skip or restart earlier items).
- Persists per-sector results so runs can be resumed without redoing answered items.
- Computes Accuracy (%) for the multiple-choice VQA tasks.
- Runs zero-shot, few-shot and chain-of-thought (cot) modes for all sectors (outputs separate files per mode).
- Sample run uses 50 examples per sector by default.
"""

import os
import json
import time
import re
from pathlib import Path
from tqdm import tqdm
import google.generativeai as genai

# ---------------- CONFIG (KEY LIST) ----------------
KEY_LIST = [
    # Add your Gemini API keys here
]

MODEL = "gemini-2.0-flash"
IMG_EXTS = {".png", ".jpg", ".jpeg"}

OUTPUT_ROOT = Path(r"...\gemini_2.0_vqa")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

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

PROMPT_CHAIN_OF_THOUGHTS = (
    "You are an AI assistant that answers visual multiple-choice questions in Bangla.\n\n"
    "Task:\n"
    "1. Look carefully at the given image: {image_path}\n"
    "2. Read the question: {question}\n"
    "3. Review the provided answer choices: {options}\n"
    "4. Select the **single most accurate answer**.\n\n"
    "Response Rules:\n"
    "- The index must be the programming list index (starting from 0).\n"
    "- Use Bangla text for the answer option.\n"
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
    "Final Answer: Index: <option_index>, Answer: \"<option_text_in_Bangla>\""
)

# ----------------- Helpers -----------------

def normalize_text(text):
    import unicodedata
    import re
    if text is None:
        return ""
    t = unicodedata.normalize("NFKC", str(text))
    t = re.sub(r"\s+", " ", t).strip()
    return t

# ----------------- Robust response parsing & JSON-correct output -----------------

def _resp_to_text(resp):
    """
    Normalize various genai SDK response shapes into a single text string.
    """
    try:
        # common direct .text
        if hasattr(resp, "text") and resp.text:
            return str(resp.text).strip()
        # some SDKs: resp.candidates -> list of candidate objects with content.parts
        if hasattr(resp, "candidates") and resp.candidates:
            texts = []
            for c in resp.candidates:
                # candidate may be a rich object
                try:
                    part_text = c.content.parts[0].text
                except Exception:
                    try:
                        part_text = c.get("content", {}).get("parts", [{}])[0].get("text", "")
                    except Exception:
                        part_text = str(c)
                if part_text:
                    texts.append(str(part_text).strip())
            if texts:
                return "\n\n---\n\n".join(texts).strip()
        # fallback to stringifying object
        return str(resp).strip()
    except Exception as e:
        return f"⚠️ Resp->text failure: {e}"


def extract_index_from_answer(answer_text):
    """
    Extract predicted index, reasoning steps, and predicted answer string from model output.

    Returns tuple:
       (pred_idx_or_None, reasoning_steps_list_or_None, reasoning_text_or_None, predicted_answer_text_or_None)
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
    m_ans = re.search(r'Answer\s*[:\-]?\s*["“]?([^"”\n]+)["”]?', text, flags=re.IGNORECASE)
    if m_ans:
        predicted_answer_text = m_ans.group(1).strip()

    # 6) Also attempt to extract reasoning_text more generically if not captured above
    reasoning_text = None
    if reasoning_block:
        reasoning_text = reasoning_block
    else:
        m_reason2 = re.search(r'Reasoning\s*[:\-]\s*(.*?)(?:Final\s*Answer|Index\s*:|\Z)', text, flags=re.IGNORECASE | re.DOTALL)
        reasoning_text = m_reason2.group(1).strip() if m_reason2 else None
        if reasoning_text:
            lines = [ln.strip() for ln in reasoning_text.splitlines() if ln.strip()]
            reasoning_steps = lines if lines else reasoning_steps

    return idx, reasoning_steps, reasoning_text, predicted_answer_text

# ----------------- Rotating Gemini VQA client -----------------
class RotatingGeminiVQA:
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

    def ask_once(self, image_path, question, options, prompt_mode="few"):
        # choose prompt
        if prompt_mode == "zero":
            prompt = PROMPT_ZERO_SHOT.format(image_path=str(image_path), question=question, options=json.dumps(options, ensure_ascii=False))
        elif prompt_mode == "cot":
            prompt = PROMPT_CHAIN_OF_THOUGHTS.format(image_path=str(image_path), question=question, options=json.dumps(options, ensure_ascii=False))
        else:
            prompt = PROMPT_FEW_SHOT.format(image_path=str(image_path), question=question, options=json.dumps(options, ensure_ascii=False))

        # upload image and call model
        img = genai.upload_file(str(image_path))
        model = genai.GenerativeModel(self.model_name)

        # generate content: defend against different SDK call shapes
        try:
            resp = model.generate_content([prompt, img])
        except TypeError:
            # attempt alternate call signature
            try:
                resp = model.generate([prompt, img])
            except Exception as e:
                raise

        # Normalize to text
        text = _resp_to_text(resp)
        return text

    def ask_with_rotation(self, image_path, question, options, prompt_mode="few", max_attempts_per_example=10, backoff_base=1.0):
        attempt = 0
        last_exc = None
        while attempt < max_attempts_per_example:
            attempt += 1
            try:
                return self.ask_once(image_path, question, options, prompt_mode=prompt_mode)
            except Exception as e:
                last_exc = e
                msg = str(e).lower()
                if "429" in msg or "quota" in msg or "rate limit" in msg or "quota exceeded" in msg:
                    print(f"❗ Quota/rate-limit detected on key index {self.key_index}: {e}")
                    self._advance_key()
                    time.sleep(1.0)
                    continue
                print(f"❗ API call failed on key index {self.key_index} (attempt {attempt}/{max_attempts_per_example}): {e}")
                self._advance_key()
                sleep_time = backoff_base * (2 ** (attempt - 1))
                print(f"⏳ backing off for {sleep_time:.1f}s before retrying this example")
                time.sleep(min(sleep_time, 60))
                continue

        print(f"❌ Failed to answer for {image_path.name} after {max_attempts_per_example} attempts. Last error: {last_exc}")
        return "❌ Failed to answer"

# ----------------- I/O helpers (persistence & files) -----------------

def load_vqa_annotations(path: Path):
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

# ----------------- Processing logic with resume support -----------------

def process_sector(sector_name, sector_cfg, key_list, prompt_mode="few", n_examples=50):
    print(f"\n==== Processing sector: {sector_name} (prompt_mode={prompt_mode}) ====")
    images_dir = sector_cfg["images"]
    annotation_file = sector_cfg["annotation"]

    vqa_data = load_vqa_annotations(annotation_file)
    vqa_data = vqa_data[:n_examples]

    if not vqa_data:
        print("⚠️ No VQA data for this sector.")
        return

    out_file = OUTPUT_ROOT / f"{sector_name}_vqa_gemini_{prompt_mode}.json"
    results_list, processed_ids = load_existing_results(out_file)

    client = RotatingGeminiVQA(key_list, MODEL)
    preds, gts = [], []

    examples_to_process = [item for item in vqa_data if item.get("image_id") not in processed_ids]
    print(f"Total examples: {len(vqa_data)}; already done: {len(processed_ids)}; to process: {len(examples_to_process)}")

    if not examples_to_process:
        print("✅ Nothing to process; computing metrics from existing results.")
        for r in results_list:
            preds.append(r.get("predicted_index"))
            gts.append(r.get("ground_truth_index"))
        acc, valid_count, correct = compute_accuracy(preds, gts)
        metrics = {"Accuracy (%)": acc, "n_examples_total": len(preds), "n_valid_evaluated": valid_count, "n_correct": correct}
        metrics_out = OUTPUT_ROOT / f"{sector_name}_vqa_metrics_gemini_{prompt_mode}.json"
        with open(metrics_out, "w", encoding="utf8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        return

    for item in tqdm(examples_to_process, desc=f"VQA ({sector_name})"):
        image_id = item.get("image_id")
        question = item.get("question")
        options = item.get("options")
        answer = item.get("answer")

        # ground truth index
        try:
            gt_index = options.index(answer)
        except Exception:
            gt_index = None

        # locate image
        img_path = None
        for ext in (".png", ".jpg", ".jpeg"):
            candidate = images_dir / f"{image_id}{ext}"
            if candidate.exists():
                img_path = candidate
                break
        if img_path is None:
            print(f"⚠️ Image not found for {image_id}")
            continue

        pred_text = client.ask_with_rotation(img_path, question, options, prompt_mode=prompt_mode, max_attempts_per_example=10)

        pred_idx, reasoning_steps, reasoning_text, predicted_answer_text = extract_index_from_answer(pred_text)

        # If the model returned a predicted answer text but no index, try to map it to options (exact normalized match)
        if pred_idx is None and predicted_answer_text:
            try:
                norm_pred = normalize_text(predicted_answer_text).lower()
                mapped = None
                for i, opt in enumerate(options):
                    if normalize_text(opt).lower() == norm_pred:
                        mapped = i
                        break
                pred_idx = mapped
            except Exception:
                pred_idx = None

        # Compose standardized result item (keeps original fields and adds reasoning fields)
        result_item = {
            "image_id": str(image_id),
            "question": normalize_text(question),
            "options": [normalize_text(x) for x in options],
            "ground_truth_index": int(gt_index) if gt_index is not None else None,
            "ground_truth_answer": normalize_text(answer) if answer is not None else None,
            "predicted_index": int(pred_idx) if pred_idx is not None else None,
            # For backward compatibility keep your "answer_text" field like your example
            # "answer_text": normalize_text(pred_text),
            # "predicted_answer_text": normalize_text(predicted_answer_text) if predicted_answer_text else None,
            # "reasoning_text": reasoning_text if reasoning_text else None,
            "reasoning_steps_en": reasoning_steps if reasoning_steps else None,
        }

        results_list.append(result_item)
        save_results_atomic(out_file, results_list)

        preds.append(pred_idx)
        gts.append(gt_index)

        # polite pause (tune as needed)
        time.sleep(5)

    acc, valid_count, correct = compute_accuracy(preds, gts)
    # count how many records included reasoning
    reasoning_count = sum(1 for r in results_list if r.get("reasoning_text"))
    metrics = {"Accuracy (%)": acc, "n_examples_total": len(preds), "n_valid_evaluated": valid_count, "n_correct": correct, "n_with_reasoning": reasoning_count}
    metrics_out = OUTPUT_ROOT / f"{sector_name}_vqa_metrics_gemini_{prompt_mode}.json"
    with open(metrics_out, "w", encoding="utf8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"✅ Finished sector {sector_name}. Results: {out_file}, Metrics: {metrics_out}")


def main():
    # run three prompt modes: zero-shot, few-shot, and chain-of-thought (cot)
    for prompt_mode in ["cot"]:
        print(f"\n########### Starting run for prompt_mode={prompt_mode} ###########")
        for sector_name, cfg in SECTORS.items():
            try:
                # sample run uses 50 examples per sector
                process_sector(sector_name, cfg, KEY_LIST, prompt_mode=prompt_mode, n_examples=50)
            except Exception as e:
                print(f"❗ Error processing sector {sector_name} ({prompt_mode}-shot): {e}")

# def main():
#     # run only zero-shot and process these sectors in this exact order
#     prompt_mode = "few"
#     desired_sectors = ["politics", "sports"]
#     missing = [s for s in desired_sectors if s not in SECTORS]
#     if missing:
#         print(f"⚠️ The following desired sectors are missing from SECTORS and will be skipped: {missing}")
#     ordered_sectors = [s for s in desired_sectors if s in SECTORS]

#     print(f"\n########### Starting run for prompt_mode={prompt_mode} ###########")
#     for sector_name in ordered_sectors:
#         cfg = SECTORS[sector_name]
#         try:
#             # sample run uses 50 examples per sector
#             process_sector(sector_name, cfg, KEY_LIST, prompt_mode=prompt_mode, n_examples=50)
#         except Exception as e:
#             print(f"❗ Error processing sector {sector_name} ({prompt_mode}-shot): {e}")



if __name__ == "__main__":
    main()

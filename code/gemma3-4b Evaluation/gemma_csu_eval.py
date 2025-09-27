#!/usr/bin/env python3
"""
csu_ollama_only.py

- Uses only a local Ollama/Gemma model to answer CSU image+question examples.
- Per-sector resume-capable processing, per-example persistence.
- Computes Exact Match (%) and BERTScore_F1 (rescale_with_baseline True with fallback).
- Prompts instruct the model to respond in Bangla only.
"""

import os
import json
import time
import re
from pathlib import Path
from tqdm import tqdm

# third-party libs
from bert_score import score as bertscore_score

# Optional: ollama client import. If you don't have it, pip-install appropriate client or adapt.
try:
    from ollama import Client as OllamaClient
except Exception:
    OllamaClient = None

# ---------------- CONFIG (OLLAMA) ----------------
MODE = "ollama"  # kept as indicator; only 'ollama' supported in this script

LLM_URL = "http://localhost:11434"   # change if your Ollama server is elsewhere
LLM_NAME = "gemma3:4b"               # change to your local model name
LLM_NUM_CTX = 4096
LLM_SEED = 0

# Model call settings
MAX_ATTEMPTS_PER_EXAMPLE = 5
SLEEP_BETWEEN_EXAMPLES = 5  # seconds (adjust if you see issues)

# Image extensions
IMG_EXTS = {".png", ".jpg", ".jpeg"}

# Output root (change with your local directory)
OUTPUT_ROOT = Path(r"...\gemma_csu")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# SECTORS
SECTORS = {
    "culture": {
        "images": Path(
            r"...\data\culture\images"
        ),
        "annotation": Path(
            r"...\data\culture\annotations\culture_commonsense_reasoning.json"
        ),
    },
    "food": {
        "images": Path(
            r"...\data\food\images"
        ),
        "annotation": Path(
            r"...\data\food\annotations\food_commonsense_reasoning.json"
        ),
    },
    "history": {
        "images": Path(
            r"...\data\history\images"
        ),
        "annotation": Path(
            r"...\data\history\annotations\history_commonsense_reasoning.json"
        ),
    },
    "media_and_movies": {
        "images": Path(
            r"...\data\media_and_movies\images"
        ),
        "annotation": Path(
            r"...\data\media_and_movies\annotations\media_and_movies_commonsense_reasoning.json"
        ),
    },
    "national_achievements": {
        "images": Path(
            r"...\data\national_achievements\images"
        ),
        "annotation": Path(
            r"...\data\national_achievements\annotations\national_achievements_commonsense_reasoning.json"
        ),
    },
    "nature": {
        "images": Path(
            r"...\data\nature\images"
        ),
        "annotation": Path(
            r"...\data\nature\annotations\nature_commonsense_reasoning.json"
        ),
    },
    "personalities": {
        "images": Path(
            r"...\data\personalities\images"
        ),
        "annotation": Path(
            r"...\data\personalities\annotations\personalities_commonsense_reasoning.json"
        ),
    },
    "politics": {
        "images": Path(
            r"...\data\politics\images"
        ),
        "annotation": Path(
            r"...\data\politics\annotations\politics_commonsense_reasoning.json"
        ),
    },
    "sports": {
        "images": Path(
            r"...\data\sports\images"
        ),
        "annotation": Path(
            r"...\data\sports\annotations\sports_commonsense_reasoning.json"
        ),
    },
}

# Prompts
PROMPT_ZERO_SHOT = (
    "You are an expert assistant for Bangla culture and commonsense reasoning tasks. "
    "You are given an image and a question. Carefully look at the image and answer the question.\n\n"
    "RESPONSE RULES (VERY IMPORTANT):\n"
    "1) Your answer MUST be written **only in Bangla** (Bangla script).\n"
    "2) Use Bangla digits for numbers if possible.\n"
    "3) Keep the answer short, direct, and on a single line (no extra explanation).\n"
    "4) Do NOT include any English words, labels, quotes, explanations, or metadata.\n"
    "5) If you are not sure about the answer, respond exactly with: অনুমান করা যাচ্ছে না\n\n"
    "Image: {image_path}\n"
    "Question: {question}\n"
    "Answer:"
)

PROMPT_FEW_SHOT = (
    "You are an expert assistant for Bangla culture and commonsense reasoning tasks. "
    "You are given an image and a question. Carefully look at the image and answer the question.\n\n"
    "RESPONSE RULES (VERY IMPORTANT):\n"
    "1) Your answer MUST be written **only in Bangla** (Bangla script).\n"
    "2) Use Bangla digits for numbers if possible.\n"
    "3) Keep the answer short, direct, and on a single line (no extra explanation).\n"
    "4) Do NOT include any English words, labels, quotes, explanations, or metadata.\n"
    "5) If you are not sure about the answer, respond exactly with: অনুমান করা যাচ্ছে না\n\n"
    "EXAMPLES:\n"
    "Image: ./dataset/history/images/history_002.png\n"
    "Question: \"ছবিতে কী হচ্ছে?\"\n"
    "Answer: বাংলাদেশ মুক্তিযুদ্ধের আত্মসমর্পণ দৃশ্য\n\n"
    "Image: ./dataset/culture/images/culture_003.png\n"
    "Question: \"ছবির এই দিনে মানুষ কেন ফুল দিয়ে সাজে?\"\n"
    "Answer: পহেলা ফাল্গুন উদযাপন\n\n"
    "Image: ./dataset/sports/images/sports_001.png\n"
    "Question: \"ছবিতে সাকিব আল হাসান কিসের জন্য বিখ্যাত?\"\n"
    "Answer: অলরাউন্ডার হিসেবে খ্যাত\n\n"
    "Now, answer for the given image strictly following the above rules.\n\n"
    "Image: {image_path}\n"
    "Question: {question}\n"
    "Answer:"
)

# ---------------- Helpers ----------------

def normalize_text(text):
    import unicodedata, re
    if text is None:
        return ""
    t = unicodedata.normalize("NFKC", str(text))
    t = re.sub(r"\s+", " ", t).strip()
    return t

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

# ---------- metrics ----------

def compute_metrics(preds, refs):
    em_scores = [int(normalize_text(p) == normalize_text(r)) for p, r in zip(preds, refs)]
    em = round(100.0 * sum(em_scores) / len(em_scores), 2) if em_scores else 0.0

    bertscore = None
    try:
        P, R, F1 = bertscore_score(preds, refs, lang="bn", rescale_with_baseline=True)
        try:
            import numpy as _np
            bertscore = float(_np.mean(F1))
        except Exception:
            import torch as _torch
            if isinstance(F1, _torch.Tensor):
                bertscore = float(F1.mean().item())
            else:
                bertscore = float(sum([float(x) for x in F1]) / len(F1))
    except Exception as e:
        print(f"⚠️ BERTScore rescale failed: {e}. Falling back to raw BERTScore.")
        try:
            P, R, F1 = bertscore_score(preds, refs, lang="bn", rescale_with_baseline=False)
            import numpy as _np
            bertscore = float(_np.mean(F1))
        except Exception as e2:
            print(f"❗ BERTScore computation failed entirely: {e2}")
            bertscore = None

    return {"Exact Match (%)": em, "BERTScore_F1": bertscore, "n_examples": len(preds)}

# ---------------- LLM Wrapper (Ollama) ----------------

class OllamaLLM:
    def __init__(self, llm_url: str, llm_name: str, llm_num_ctx: int = 4096, llm_seed: int = 0):
        if OllamaClient is None:
            raise RuntimeError("Ollama client not available. Install the Ollama client or adapt this script.")
        self.llm_name = llm_name
        self.llm_client = OllamaClient(host=llm_url)
        self.llm_num_ctx = llm_num_ctx
        self.llm_seed = llm_seed

    def generate(self, prompt: str, image: str = None, max_tokens: int = 256):
        generate_args = {
            "model": self.llm_name,
            "prompt": prompt,
            "options": {
                "seed": self.llm_seed,
                "num_ctx": self.llm_num_ctx,
                "num_predict": max_tokens,
            },
        }
        if image:
            generate_args["images"] = [image]

        response = self.llm_client.generate(**generate_args)
        # Extract ONLY the response string
        if isinstance(response, dict):
            resp_text = response.get("response") or response.get("output") or ""
        else:
            resp_text = str(response)
        # regex extraction: response='...' or output='...'
        m = re.search(r"(?:response|output)='(.*?)'", resp_text)
        if m:
            resp_text = m.group(1)
        return resp_text.strip()

# ----------------- Main processing -----------------

def process_sector_with_ollama(sector_name, sector_cfg, llm, prompt_mode="few", n_examples=10):
    print(f"\n==== Processing sector: {sector_name} (prompt_mode={prompt_mode}) ====")
    images_dir = sector_cfg["images"]
    annotation_file = sector_cfg["annotation"]

    csu_data = load_csu_annotations(annotation_file)
    csu_data = csu_data[:n_examples]

    if not csu_data:
        print("⚠️ No CSU data for this sector.")
        return

    out_file = OUTPUT_ROOT / f"{sector_name}_csu_results_ollama_{prompt_mode}.json"
    results_list, processed_ids = load_existing_results(out_file)

    preds, refs = [], []

    examples_to_process = [item for item in csu_data if item.get("image_id") not in processed_ids]
    print(f"Total examples: {len(csu_data)}; already done: {len(processed_ids)}; to process: {len(examples_to_process)}")

    if not examples_to_process:
        print("✅ Nothing to process; computing metrics from existing results.")
        for r in results_list:
            preds.append(r.get("predicted_answer", ""))
            refs.append(r.get("ground_truth_answer", ""))
        metrics = compute_metrics(preds, refs) if refs else {"error": "no_examples"}
        metrics_out = OUTPUT_ROOT / f"{sector_name}_csu_metrics_ollama_{prompt_mode}.json"
        with open(metrics_out, "w", encoding="utf8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print("Saved metrics to:", metrics_out)
        return

    for item in tqdm(examples_to_process, desc=f"CSU ({sector_name})"):
        image_id = item.get("image_id")
        question = item.get("question")
        answer = item.get("answer")

        # find image
        img_path = None
        for ext in IMG_EXTS:
            cand = images_dir / f"{image_id}{ext}"
            if cand.exists():
                img_path = cand
                break
        if img_path is None:
            print(f"⚠️ Image not found for {image_id}")
            continue

        # build prompt
        prompt = PROMPT_ZERO_SHOT.format(image_path=str(img_path), question=question) if prompt_mode=="zero" else PROMPT_FEW_SHOT.format(image_path=str(img_path), question=question)

        # generate answer
        pred_answer = None
        attempts = 0
        last_exc = None
        while attempts < MAX_ATTEMPTS_PER_EXAMPLE:
            attempts += 1
            try:
                pred_answer = llm.generate(prompt, image=str(img_path), max_tokens=256)
                break
            except Exception as e:
                last_exc = e
                print(f"❗ Ollama generation failed for {image_id} (attempt {attempts}): {e}")
                time.sleep(min(2 ** attempts, 30))
        if pred_answer is None:
            pred_answer = "❌ Failed to answer"
            print(f"❌ Generation failed for {image_id} after {MAX_ATTEMPTS_PER_EXAMPLE} attempts. Last error: {last_exc}")

        preds.append(pred_answer)
        refs.append(answer)
        results_list.append({
            "image_id": image_id,
            "question": question,
            "ground_truth_answer": answer,
            "predicted_answer": pred_answer,
        })

        save_results_atomic(out_file, results_list)
        time.sleep(SLEEP_BETWEEN_EXAMPLES)

    metrics = compute_metrics(preds, refs)
    metrics_out = OUTPUT_ROOT / f"{sector_name}_csu_metrics_ollama_{prompt_mode}.json"
    with open(metrics_out, "w", encoding="utf8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"✅ Finished sector {sector_name}. Results -> {out_file}; Metrics -> {metrics_out}")

# ----------------- Entrypoint -----------------

def main():
    if OllamaClient is None:
        raise RuntimeError("Ollama client not found. Install the Ollama client Python package or adapt this script.")
    llm = OllamaLLM(LLM_URL, LLM_NAME, llm_num_ctx=LLM_NUM_CTX, llm_seed=LLM_SEED)
    print(f"Using Ollama model {LLM_NAME} at {LLM_URL}")

    for prompt_mode in ["zero", "few"]:
        print(f"\n==== Running prompt_mode: {prompt_mode} ====")
        for sector_name, cfg in SECTORS.items():
            try:
                process_sector_with_ollama(sector_name, cfg, llm, prompt_mode=prompt_mode, n_examples=None)
            except Exception as e:
                print(f"❗ Error processing sector {sector_name} ({prompt_mode}): {e}")

if __name__ == "__main__":
    main()

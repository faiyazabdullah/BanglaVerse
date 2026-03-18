#!/usr/bin/env python3
"""Shared utilities for caption and VQA evaluation scripts."""

import os
import json
import time
import re
import random
import base64
import io
import shutil
import unicodedata
from pathlib import Path

from tqdm import tqdm
import numpy as np
import sacrebleu
from openai import OpenAI
from bert_score import score as bertscore_score
from PIL import Image

# ---- Config ----
MODEL_NAME = "gpt-4.1-mini"
_api_key = os.environ.get("OPENAI_API_KEY")
if not _api_key:
    raise RuntimeError(
        "OPENAI_API_KEY environment variable is not set. Export it before running: export OPENAI_API_KEY=sk-..."
    )
KEY_LIST = [_api_key]
MAX_SAMPLES = 100
MAX_IMAGE_SIZE = 1024
IMG_EXTS = {".png", ".jpg", ".jpeg"}
N_JOBS = 8

WORKSPACE_ROOT = Path(__file__).resolve().parent
DATA_ROOT = WORKSPACE_ROOT / "Data"
RESULTS_ROOT = WORKSPACE_ROOT / "Results"
EVAL_DATA_ROOT = WORKSPACE_ROOT / "eval_data"

SECTOR_NAMES = [
    "culture",
    "food",
    "history",
    "media_and_movies",
    "national_achievements",
    "nature",
    "personalities",
    "politics",
    "sports",
]

# ---- Image helpers ----


def resize_image_if_needed(image_path, max_size=MAX_IMAGE_SIZE):
    """Resize image if any dimension exceeds max_size, maintaining aspect ratio."""
    try:
        img = Image.open(image_path)
        if img.mode == "RGBA":
            rgb_img = Image.new("RGB", img.size, (255, 255, 255))
            rgb_img.paste(img, mask=img.split()[3])
            img = rgb_img
        elif img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        if max(img.size) > max_size:
            img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=95)
        return buffer.getvalue()
    except Exception as e:
        print(f"Warning: Error resizing image {image_path}: {e}")
        with open(image_path, "rb") as f:
            return f.read()


def encode_image_base64(image_path):
    """Encode image to base64 string with resizing."""
    image_bytes = resize_image_if_needed(image_path)
    return base64.b64encode(image_bytes).decode("utf-8")


def get_image_mime_type(image_path):
    """Get MIME type - always JPEG after resizing."""
    return "image/jpeg"


def find_image(folder, image_id):
    """Find an image file by ID, trying all supported extensions."""
    for ext in IMG_EXTS:
        candidate = folder / f"{image_id}{ext}"
        if candidate.exists():
            return candidate
    return None


def list_images(folder):
    """List image files in a folder, sorted by name."""
    if not folder.exists():
        print(f"Warning: Image folder not found: {folder}")
        return []
    return sorted([p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS])


# ---- Persistence helpers ----


def save_json_atomic(out_file, data):
    """Atomically write JSON data to a file."""
    tmp = out_file.with_suffix(out_file.suffix + ".tmp")
    with open(tmp, "w", encoding="utf8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp.replace(out_file)


# ---- Caption helpers ----


def load_gold_annotations(path):
    if not path.exists():
        print(f"Warning: Annotation file not found: {path}")
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
        print("Warning: Unknown annotation format; expected list or dict.")
        return {}


def get_random_caption_examples(gold_dict, current_image_id, n=3):
    """Select n random caption examples, excluding current_image_id."""
    available = [
        (img_id, cap) for img_id, cap in gold_dict.items() if img_id != current_image_id
    ]
    if len(available) <= n:
        return available
    return random.sample(available, n)


def format_caption_examples(examples):
    """Format caption examples for the prompt."""
    formatted = []
    for img_id, caption in examples:
        formatted.append(f"Image ID: {img_id}\nCaption: {caption}")
    return "\n\n".join(formatted)


# ---- Metrics ----


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
    """Calculate ROUGE-L F1 (character-level)."""
    lcs = lcs_length(pred, ref)
    precision = lcs / len(pred) if len(pred) > 0 else 0
    recall = lcs / len(ref) if len(ref) > 0 else 0
    if precision + recall == 0:
        return 0.0
    return (2 * precision * recall) / (precision + recall)


def compute_metrics_for_sector(preds_dict, gold_dict, lang="en"):
    """Compute BLEU, ROUGE-L, and BertScore F1 for a sector."""
    preds, refs, skipped = [], [], []
    for imgid, pred in preds_dict.items():
        gold = gold_dict.get(imgid)
        if gold and pred and not pred.startswith("❌") and not pred.startswith("⚠️"):
            preds.append(pred)
            refs.append(gold)
        else:
            skipped.append(imgid)

    metrics = {"n_images_with_gold": len(preds), "n_skipped": len(skipped)}
    if len(preds) == 0:
        return metrics

    # BLEU
    try:
        bleu = sacrebleu.corpus_bleu(preds, [refs])
        metrics["BLEU"] = float(bleu.score)
    except Exception as e:
        print("BLEU calc failed:", e)
        metrics["BLEU"] = None

    # ROUGE-L via LCS
    try:
        rouge_l_scores = [compute_rouge_l(p, r) for p, r in zip(preds, refs)]
        metrics["ROUGE-L"] = float(np.mean(rouge_l_scores))
    except Exception as e:
        print("ROUGE-L calc failed:", e)
        metrics["ROUGE-L"] = None

    # BertScore
    try:
        P, R, F1 = bertscore_score(preds, refs, lang=lang, rescale_with_baseline=True)
        try:
            bert_f1 = float(np.mean(F1))
        except Exception:
            try:
                import torch

                if isinstance(F1, torch.Tensor):
                    bert_f1 = float(F1.mean().item())
                else:
                    bert_f1 = float(np.mean(F1))
            except Exception:
                bert_f1 = float(np.mean([float(x) for x in F1]))
        metrics["BertScore_F1"] = bert_f1
    except Exception as e:
        print(f"Warning: BERTScore rescale failed: {e}. Falling back to raw.")
        try:
            P, R, F1 = bertscore_score(
                preds, refs, lang=lang, rescale_with_baseline=False
            )
            try:
                bert_f1_raw = float(np.mean(F1))
            except Exception:
                import torch

                if isinstance(F1, torch.Tensor):
                    bert_f1_raw = float(F1.mean().item())
                else:
                    bert_f1_raw = float(np.mean(F1))
            metrics["BertScore_F1"] = bert_f1_raw
            metrics["BertScore_F1_raw"] = bert_f1_raw
        except Exception as e2:
            print(f"Error: BERTScore computation failed entirely: {e2}")
            metrics["BertScore_F1"] = None
            metrics["BertScore_F1_raw"] = None

    return metrics


# ---- VQA helpers ----


def normalize_text(text):
    if text is None:
        return ""
    t = unicodedata.normalize("NFKC", str(text))
    t = re.sub(r"\s+", " ", t).strip()
    return t


def extract_index_from_answer(answer_text):
    """
    Extract predicted index, reasoning steps, and predicted answer string from model output.
    Returns (pred_idx, reasoning_steps, reasoning_text, predicted_answer_text).
    """
    if not answer_text:
        return None, None, None, None

    text = str(answer_text).strip()

    def _clean_step(s):
        return re.sub(r"^\s*Step\s*\d+\s*[:\-]?\s*", "", s, flags=re.IGNORECASE).strip()

    # 1) Try to find Reasoning_En block then Final Answer (CoT)
    reasoning_block = None
    m_reason = re.search(
        r"Reasoning[_ ]?En\s*[:\-]?\s*(.*?)(?:Final\s*Answer|Index\s*:|\Z)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if m_reason:
        reasoning_block = m_reason.group(1).strip()

    reasoning_steps = None
    if reasoning_block:
        steps = []
        for i in range(1, 9):
            m_step = re.search(
                r"(?:^|\n)\s*Step\s*"
                + str(i)
                + r"\s*[:\-]?\s*(.*?)(?=(?:\n\s*Step\s*"
                + str(i + 1)
                + r"\b)|\Z)",
                reasoning_block,
                flags=re.IGNORECASE | re.DOTALL,
            )
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

    # 2) Extract index via "Final Answer: Index: X"
    m_final_idx = re.search(
        r"Final\s*Answer\s*[:\-]?\s*(?:Index\s*[:\-]?\s*(\d+))",
        text,
        flags=re.IGNORECASE,
    )
    if m_final_idx:
        idx = int(m_final_idx.group(1))
    else:
        m_idx_any = re.search(r"\bIndex\s*[:\-]?\s*(\d+)\b", text, flags=re.IGNORECASE)
        idx = int(m_idx_any.group(1)) if m_idx_any else None

    # 3) Extract predicted answer text
    predicted_answer_text = None
    m_final_ans = re.search(
        r'Final\s*Answer\s*[:\-]?.*?Answer\s*[:\-]?\s*[""\u201c]?([^""\u201d\n]+)[""\u201d]?',
        text,
        flags=re.IGNORECASE,
    )
    if m_final_ans:
        predicted_answer_text = m_final_ans.group(1).strip()
    else:
        m_ans = re.search(
            r'Answer\s*[:\-]?\s*[""\u201c]?([^""\u201d\n]+)[""\u201d]?',
            text,
            flags=re.IGNORECASE,
        )
        if m_ans:
            predicted_answer_text = m_ans.group(1).strip()

    # 4) Extract reasoning_text generically if not captured above
    reasoning_text = None
    if reasoning_block:
        reasoning_text = reasoning_block
    else:
        m_reason2 = re.search(
            r"Reasoning\s*[:\-]\s*(.*?)(?:Final\s*Answer|Index\s*:|\Z)",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        reasoning_text = m_reason2.group(1).strip() if m_reason2 else None
        if reasoning_text:
            lines = [ln.strip() for ln in reasoning_text.splitlines() if ln.strip()]
            reasoning_steps = lines if lines else reasoning_steps

    return idx, reasoning_steps, reasoning_text, predicted_answer_text


def load_vqa_annotations(path):
    if not path.exists():
        print(f"Warning: Annotation file not found: {path}")
        return []
    with open(path, "r", encoding="utf8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else []


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
        img_path = None
        for ext in (".png", ".jpg", ".jpeg"):
            candidate = images_dir / f"{image_id}{ext}"
            if candidate.exists():
                img_path = candidate
                break
        if img_path:
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
                examples_with_images.append(
                    {
                        "image_id": image_id,
                        "image_path": img_path,
                        "question": item.get("question"),
                        "options": options,
                        "answer_index": gt_index,
                        "answer_text": options[gt_index]
                        if 0 <= gt_index < len(options)
                        else answer,
                    }
                )

    return examples_with_images


def format_vqa_examples(examples):
    """Format the VQA examples for the few-shot prompt."""
    formatted = []
    for ex in examples:
        formatted.append(
            f'Question: "{ex["question"]}"\n'
            f"Options: {json.dumps(ex['options'], ensure_ascii=False)}\n"
            f'Answer: Index: {ex["answer_index"]}, Answer: "{ex["answer_text"]}"\n'
        )
    return "\n".join(formatted)


def compute_accuracy(preds, gts):
    """Compute accuracy on pairs where both pred and gt are not None."""
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


# ---- API clients ----


class RotatingGPTCaptioner:
    def __init__(self, key_list):
        assert key_list, "Provide at least one API key"
        self.keys = key_list
        self.key_index = 0
        self._configure_current_key()

    def _configure_current_key(self):
        key = self.keys[self.key_index]
        self.client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key)
        print(f"Using API key index {self.key_index}")

    def _advance_key(self):
        old = self.key_index
        self.key_index = (self.key_index + 1) % len(self.keys)
        print(f"Switching API key: {old} -> {self.key_index}")
        self._configure_current_key()

    def generate_caption_once(self, image_path, prompt_text, extra_images=None):
        """Single attempt. Returns (caption, input_tokens, output_tokens)."""
        content = [
            {"type": "text", "text": prompt_text},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{get_image_mime_type(image_path)};base64,{encode_image_base64(image_path)}"
                },
            },
        ]
        if extra_images:
            for extra_img_path in extra_images:
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{get_image_mime_type(extra_img_path)};base64,{encode_image_base64(extra_img_path)}"
                        },
                    }
                )

        response = self.client.chat.completions.create(
            messages=[{"role": "user", "content": content}],
            max_completion_tokens=16384,
            model=MODEL_NAME,
        )
        input_tokens = (
            response.usage.prompt_tokens
            if hasattr(response, "usage") and response.usage
            else 0
        )
        output_tokens = (
            response.usage.completion_tokens
            if hasattr(response, "usage") and response.usage
            else 0
        )
        caption = (
            response.choices[0].message.content.strip()
            if response.choices
            else "Warning: Empty response"
        )
        return caption, input_tokens, output_tokens

    def generate_caption_with_rotation(
        self, image_path, prompt_text, extra_images=None, backoff_base=1.0
    ):
        """Retry indefinitely with key rotation and exponential backoff."""
        attempt = 0
        while True:
            attempt += 1
            try:
                return self.generate_caption_once(image_path, prompt_text, extra_images)
            except (FileNotFoundError, PermissionError, ValueError, TypeError) as e:
                # Permanent errors - don't retry
                print(f"Permanent error: {e}")
                raise
            except Exception as e:
                msg = str(e).lower()
                if (
                    "429" in msg
                    or "quota" in msg
                    or "rate limit" in msg
                    or "quota exceeded" in msg
                ):
                    print(f"Quota/rate-limit on key {self.key_index}: {e}")
                    self._advance_key()
                    time.sleep(1.0)
                    continue
                print(
                    f"API call failed on key {self.key_index} (attempt {attempt}): {e}"
                )
                self._advance_key()
                sleep_time = backoff_base * (2 ** (attempt - 1))
                print(f"Backing off for {sleep_time:.1f}s")
                time.sleep(min(sleep_time, 120))
                continue


class RotatingGPTVQA:
    def __init__(self, key_list):
        assert key_list, "Provide at least one API key"
        self.keys = key_list
        self.key_index = 0
        self._configure_current_key()

    def _configure_current_key(self):
        key = self.keys[self.key_index]
        self.client = OpenAI(api_key=key, base_url="https://openrouter.ai/api/v1")
        print(f"Using API key index {self.key_index}")

    def _advance_key(self):
        old = self.key_index
        self.key_index = (self.key_index + 1) % len(self.keys)
        print(f"Switching API key: {old} -> {self.key_index}")
        self._configure_current_key()

    def ask_once(self, image_paths, question, options, prompt, examples_images=None):
        content = [{"type": "text", "text": prompt}]
        if examples_images:
            for img_path in examples_images:
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{get_image_mime_type(img_path)};base64,{encode_image_base64(img_path)}"
                        },
                    }
                )
        if isinstance(image_paths, list):
            for img_path in image_paths:
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{get_image_mime_type(img_path)};base64,{encode_image_base64(img_path)}"
                        },
                    }
                )
        else:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{get_image_mime_type(image_paths)};base64,{encode_image_base64(image_paths)}"
                    },
                }
            )

        response = self.client.chat.completions.create(
            messages=[{"role": "user", "content": content}],
            max_completion_tokens=16384,
            model=MODEL_NAME,
        )
        input_tokens = (
            response.usage.prompt_tokens
            if hasattr(response, "usage") and response.usage
            else 0
        )
        output_tokens = (
            response.usage.completion_tokens
            if hasattr(response, "usage") and response.usage
            else 0
        )
        text = (
            response.choices[0].message.content.strip()
            if response.choices
            else "Warning: Empty response"
        )
        return text, input_tokens, output_tokens

    def ask_with_rotation(
        self,
        image_path,
        question,
        options,
        prompt,
        examples_images=None,
        backoff_base=1.0,
    ):
        """Retry indefinitely with key rotation and exponential backoff."""
        attempt = 0
        while True:
            attempt += 1
            try:
                return self.ask_once(
                    image_path,
                    question,
                    options,
                    prompt,
                    examples_images=examples_images,
                )
            except (FileNotFoundError, PermissionError, ValueError, TypeError) as e:
                # Permanent errors - don't retry
                print(f"Permanent error: {e}")
                raise
            except Exception as e:
                msg = str(e).lower()
                if (
                    "429" in msg
                    or "quota" in msg
                    or "rate limit" in msg
                    or "quota exceeded" in msg
                ):
                    print(f"Quota/rate-limit on key {self.key_index}: {e}")
                    self._advance_key()
                    time.sleep(1.0)
                    continue
                print(
                    f"API call failed on key {self.key_index} (attempt {attempt}): {e}"
                )
                self._advance_key()
                sleep_time = backoff_base * (2 ** (attempt - 1))
                print(f"Backing off for {sleep_time:.1f}s")
                time.sleep(min(sleep_time, 120))
                continue


# ---- Eval data snapshot ----


def _snapshot_eval_images(image_paths):
    """Copy used images to eval_data/, preserving path structure relative to DATA_ROOT."""
    for img_path in image_paths:
        try:
            rel = img_path.relative_to(DATA_ROOT)
        except ValueError:
            continue
        dest = EVAL_DATA_ROOT / rel
        if dest.exists():
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(img_path, dest)


def _snapshot_eval_annotations(annotation_file, used_data):
    """Save only the used annotation entries to eval_data/, preserving path structure."""
    try:
        rel = annotation_file.relative_to(DATA_ROOT)
    except ValueError:
        return
    dest = EVAL_DATA_ROOT / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    save_json_atomic(dest, used_data)


# ---- Processing functions ----


def process_caption_sector(
    sector_name,
    sector_cfg,
    key_list,
    output_root,
    prompt_zero,
    prompt_few_template,
    bertscore_lang="en",
    prompt_mode="zero",
    n_examples=None,
):
    """Process a single sector for caption evaluation."""
    metrics_file = output_root / f"{sector_name}_metrics_gpt5_{prompt_mode}.json"
    if metrics_file.exists():
        print(
            f"Skipping {sector_name}/{prompt_mode} — already completed ({metrics_file})"
        )
        return

    print(f"\n==== Processing sector: {sector_name} (prompt_mode={prompt_mode}) ====")
    images_dir = sector_cfg["images"]
    annotation_file = sector_cfg["annotation"]

    images = list_images(images_dir)
    if n_examples:
        images = images[:n_examples]
    else:
        images = images[: min(len(images), MAX_SAMPLES)]

    if not images:
        print("Warning: No images to process in this sector.")
        return

    captioner = RotatingGPTCaptioner(key_list)
    gold = load_gold_annotations(annotation_file)

    # Snapshot used eval data (images + filtered annotations)
    used_image_ids = {img.stem for img in images}
    _snapshot_eval_images(images)
    if gold:
        used_gold = {k: v for k, v in gold.items() if k in used_image_ids}
        _snapshot_eval_annotations(annotation_file, used_gold)

    out_file = output_root / f"{sector_name}_captions_gpt5_{prompt_mode}.json"
    generated_list, generated_map = [], {}
    print(f"Total images to process: {len(images)}")

    total_input_tokens = 0
    total_output_tokens = 0

    pbar = tqdm(images, desc=f"Caption ({sector_name}/{prompt_mode})")
    for img_path in pbar:
        image_id = img_path.stem

        if prompt_mode == "zero":
            prompt_text = prompt_zero
            extra_images = None
        else:
            examples = get_random_caption_examples(gold, image_id, n=3)
            formatted_examples = format_caption_examples(examples)
            prompt_text = prompt_few_template.format(examples=formatted_examples)
            extra_images = [
                p
                for ex_id, _ in examples
                if (p := find_image(images_dir, ex_id)) is not None
            ]

        caption, input_tokens, output_tokens = captioner.generate_caption_with_rotation(
            img_path, prompt_text, extra_images=extra_images
        )

        total_input_tokens += input_tokens
        total_output_tokens += output_tokens
        pbar.set_postfix(
            tokens=f"{(total_input_tokens + total_output_tokens) / 1000:.1f}k"
        )

        entry = {
            "image_id": image_id,
            "caption": caption,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }
        generated_list.append(entry)
        generated_map[image_id] = caption
        save_json_atomic(out_file, generated_list)
        time.sleep(0.5)

    metrics = (
        compute_metrics_for_sector(generated_map, gold, lang=bertscore_lang)
        if gold
        else {"error": "no_gold_annotations"}
    )
    metrics["total_input_tokens"] = total_input_tokens
    metrics["total_output_tokens"] = total_output_tokens
    metrics["total_tokens"] = total_input_tokens + total_output_tokens

    metrics_out = output_root / f"{sector_name}_metrics_gpt5_{prompt_mode}.json"
    save_json_atomic(metrics_out, metrics)
    print(f"Finished sector {sector_name}. Metrics written to {metrics_out}")
    print(
        f"Total tokens - Input: {total_input_tokens}, Output: {total_output_tokens}, Total: {total_input_tokens + total_output_tokens}"
    )


def process_vqa_sector(
    sector_name,
    sector_cfg,
    key_list,
    output_root,
    prompt_zero,
    prompt_few_template,
    prompt_cot,
    prompt_mode="few",
    n_examples=None,
):
    """Process a single sector for VQA evaluation."""
    metrics_file = output_root / f"{sector_name}_vqa_metrics_gpt5_{prompt_mode}.json"
    if metrics_file.exists():
        print(
            f"Skipping {sector_name}/{prompt_mode} — already completed ({metrics_file})"
        )
        return

    print(f"\n==== Processing sector: {sector_name} (prompt_mode={prompt_mode}) ====")
    images_dir = sector_cfg["images"]
    annotation_file = sector_cfg["annotation"]

    vqa_data = load_vqa_annotations(annotation_file)
    if n_examples is None:
        vqa_data = vqa_data[: min(len(vqa_data), MAX_SAMPLES)]
    else:
        vqa_data = vqa_data[:n_examples]

    if not vqa_data:
        print("Warning: No VQA data for this sector.")
        return

    # Snapshot used eval data (images + filtered annotations)
    used_images = [
        p
        for item in vqa_data
        if (p := find_image(images_dir, item.get("image_id"))) is not None
    ]
    _snapshot_eval_images(used_images)
    _snapshot_eval_annotations(annotation_file, vqa_data)

    out_file = output_root / f"{sector_name}_vqa_gpt5_{prompt_mode}.json"
    results_list = []
    client = RotatingGPTVQA(key_list)
    preds, gts = [], []
    total_input_tokens = 0
    total_output_tokens = 0
    print(f"Total examples to process: {len(vqa_data)}")

    pbar = tqdm(vqa_data, desc=f"VQA ({sector_name}/{prompt_mode})")
    for item in pbar:
        image_id = item.get("image_id")
        question = item.get("question")
        options = item.get("options")
        answer = item.get("answer")

        gt_index = None
        if isinstance(answer, int):
            gt_index = answer
        elif options is not None and isinstance(options, list):
            try:
                gt_index = options.index(answer)
            except ValueError:
                pass

        img_path = None
        for ext in (".png", ".jpg", ".jpeg"):
            candidate = images_dir / f"{image_id}{ext}"
            if candidate.exists():
                img_path = candidate
                break
        if img_path is None:
            print(f"Warning: Image not found for {image_id}")
            continue

        examples_images = None
        if prompt_mode == "zero":
            prompt = prompt_zero.format(
                question=question, options=json.dumps(options, ensure_ascii=False)
            )
        elif prompt_mode == "cot":
            prompt = prompt_cot.format(
                question=question, options=json.dumps(options, ensure_ascii=False)
            )
        else:
            examples = get_random_vqa_examples(
                vqa_data, images_dir, image_id, n_examples=3
            )
            formatted_examples = format_vqa_examples(examples)
            prompt = prompt_few_template.format(
                examples=formatted_examples,
                question=question,
                options=json.dumps(options, ensure_ascii=False),
            )
            examples_images = [ex["image_path"] for ex in examples]

        pred_text, input_tokens, output_tokens = client.ask_with_rotation(
            img_path, question, options, prompt, examples_images=examples_images
        )
        total_input_tokens += input_tokens
        total_output_tokens += output_tokens
        pbar.set_postfix(
            tokens=f"{(total_input_tokens + total_output_tokens) / 1000:.1f}k"
        )

        pred_idx, reasoning_steps, reasoning_text, predicted_answer_text = (
            extract_index_from_answer(pred_text)
        )

        if pred_idx is None and predicted_answer_text:
            try:
                norm_pred = normalize_text(predicted_answer_text).lower()
                for i, opt in enumerate(options):
                    if normalize_text(opt).lower() == norm_pred:
                        pred_idx = i
                        break
            except Exception:
                pred_idx = None

        result_item = {
            "image_id": str(image_id),
            "question": normalize_text(question),
            "options": [normalize_text(x) for x in options],
            "ground_truth_index": int(gt_index) if gt_index is not None else None,
            "ground_truth_answer": normalize_text(answer)
            if answer is not None
            else None,
            "predicted_index": int(pred_idx) if pred_idx is not None else None,
            "answer_text": normalize_text(pred_text),
            "predicted_answer_text": normalize_text(predicted_answer_text)
            if predicted_answer_text
            else None,
            "reasoning_text": reasoning_text if reasoning_text else None,
            "reasoning_steps_en": reasoning_steps if reasoning_steps else None,
        }

        results_list.append(result_item)
        save_json_atomic(out_file, results_list)
        preds.append(pred_idx)
        gts.append(gt_index)
        time.sleep(5)

    acc, valid_count, correct = compute_accuracy(preds, gts)
    reasoning_count = sum(1 for r in results_list if r.get("reasoning_text"))
    metrics = {
        "Accuracy (%)": acc,
        "n_examples_total": len(preds),
        "n_valid_evaluated": valid_count,
        "n_correct": correct,
        "n_with_reasoning": reasoning_count,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_tokens": total_input_tokens + total_output_tokens,
    }
    metrics_out = output_root / f"{sector_name}_vqa_metrics_gpt5_{prompt_mode}.json"
    save_json_atomic(metrics_out, metrics)

    print(f"Finished sector {sector_name}. Results: {out_file}, Metrics: {metrics_out}")
    print(
        f"   Total tokens: {total_input_tokens + total_output_tokens} (input: {total_input_tokens}, output: {total_output_tokens})"
    )

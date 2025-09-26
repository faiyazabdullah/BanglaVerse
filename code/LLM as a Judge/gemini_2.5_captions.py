#!/usr/bin/env python3
"""
gemini_caption_judge_with_rotation.py

Resume-capable caption judge with hard-coded API key rotation.
Robust to malformed JSON entries (e.g., "food_id" instead of "image_id"),
and with option to skip sectors that are already fully judged while continuing
the overall run.
"""

import os
import json
import time
from pathlib import Path
from typing import Tuple, Optional, Dict, List

import google.generativeai as genai

# ---------------- CONFIG (HARD-CODED KEYS) ----------------
KEY_LIST = [
    # Add your Gemini API keys here
]

judge_model = "gemini-2.5-flash"

PROMPT_TEMPLATE = """You are a highly skilled and impartial caption evaluator. Your task is to carefully compare a *generated caption* with a *reference caption* and score it according to the following dimensions:

1. **Relevance (0–1):** How well does the generated caption describe the main objects, actions, and context of the reference caption? Reward high semantic overlap and penalize missing or hallucinated details.
2. **Clarity (0–1):** Is the caption grammatically correct, well-structured, and easy to read?
3. **Conciseness (0–1):** Is the caption free of redundancy, filler words, or unnecessary complexity while still conveying the full meaning?
4. **Creativity (0–1):** Does the caption show originality or an engaging phrasing, rather than being overly generic?

After scoring each dimension, compute an **Overall (0–1)** score that reflects the holistic quality of the generated caption, giving slightly higher weight to *Relevance* and *Clarity*.

Your response must strictly follow this JSON-like structure:
Relevance: [float between 0 and 1]
Clarity: [float between 0 and 1]
Conciseness: [float between 0 and 1]
Creativity: [float between 0 and 1]
Overall: [float between 0 and 1]
Explanation:
[Concise explanation: mention key strengths, weaknesses, and reasoning for the scores.]

Reference Caption: "{reference_caption}"
Generated Caption: "{generated_caption}" """

# Output root
OUTPUT_ROOT = Path(r"...\LLM as a Judge\Gemini-2.5-flash")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# SECTORS configuration (user-provided paths)
SECTORS = {
    "culture": {
        "generated_captions_zero": Path(r"...\gemini-2.5_captions\culture_captions_gemini_zero.json"),
        "generated_captions_few": Path(r"...\gemini-2.5_captions\culture_captions_gemini_few.json"),
        "annotation": Path(r"...\data\culture\annotations\culture_captions.json")
    },
    "food": {
        "generated_captions_zero": Path(r"...\gemini-2.5_captions\food_captions_gemini_zero.json"),
        "generated_captions_few": Path(r"...\gemini-2.5_captions\food_captions_gemini_few.json"),
        "annotation": Path(r"...\data\food\annotations\food_captions.json")
    },
    "history": {
        "generated_captions_zero": Path(r"...\gemini-2.5_captions\history_captions_gemini_zero.json"),
        "generated_captions_few": Path(r"...\gemini-2.5_captions\history_captions_gemini_few.json"),
        "annotation": Path(r"...\data\history\annotations\history_captions.json")
    },
    "media_and_movies": {
        "generated_captions_zero": Path(r"...\gemini-2.5_captions\media_and_movies_captions_gemini_zero.json"),
        "generated_captions_few": Path(r"...\gemini-2.5_captions\media_and_movies_captions_gemini_few.json"),
        "annotation": Path(r"...\data\media_and_movies\annotations\media_and_movies_captions.json")
    },
    "national_achievements": {
        "generated_captions_zero": Path(r"...\gemini-2.5_captions\national_achievements_captions_gemini_zero.json"),
        "generated_captions_few": Path(r"...\gemini-2.5_captions\national_achievements_captions_gemini_few.json"),
        "annotation": Path(r"...\data\national_achievements\annotations\national_achievements_captions.json")
    },
    "nature": {
        "generated_captions_zero": Path(r"...\gemini-2.5_captions\nature_captions_gemini_zero.json"),
        "generated_captions_few": Path(r"...\gemini-2.5_captions\nature_captions_gemini_few.json"),
        "annotation": Path(r"...\data\nature\annotations\nature_captions.json")
    },
    "personalities": {
        "generated_captions_zero": Path(r"...\gemini-2.5_captions\personalities_captions_gemini_zero.json"),
        "generated_captions_few": Path(r"...\gemini-2.5_captions\personalities_captions_gemini_few.json"),
        "annotation": Path(r"...\data\personalities\annotations\personalities_captions.json")
    },
    "politics": {
        "generated_captions_zero": Path(r"...\gemini-2.5_captions\politics_captions_gemini_zero.json"),
        "generated_captions_few": Path(r"...\gemini-2.5_captions\politics_captions_gemini_few.json"),
        "annotation": Path(r"...\data\politics\annotations\politics_captions.json")
    },
    "sports": {
        "generated_captions_zero": Path(r"...\gemini-2.5_captions\sports_captions_gemini_zero.json"),
        "generated_captions_few": Path(r"...\gemini-2.5_captions\sports_captions_gemini_few.json"),
        "annotation": Path(r"...\data\sports\annotations\sports_captions.json")
    }
}

# ---------------- Helpers ----------------

def load_json(path: Path):
    """Load JSON returning a list. If file missing, return [] (and warn)."""
    if not path.exists():
        print(f"⚠️ File not found: {path}")
        return []
    try:
        with open(path, "r", encoding="utf8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            print(f"⚠️ Expected a list in JSON file {path}, got {type(data)}. Returning [].")
            return []
        return data
    except Exception as e:
        print(f"❗ Failed to read/parse JSON {path}: {e}")
        return []

def save_json_atomic(path: Path, data):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp.replace(path)

def clamp_score(v: Optional[float]) -> Optional[float]:
    if v is None:
        return None
    try:
        return max(0.0, min(1.0, float(v)))
    except Exception:
        return None

def parse_judge_response(text: str) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float], Optional[float], str]:
    """Parse the expected judge response format. Returns scores (or None) and explanation."""
    import re
    if not text:
        return (None, None, None, None, None, "")
    # normalize
    t = text.replace("\r", "\n")

    def find_score(key: str):
        m = re.search(rf"{key}\s*:\s*([0-9]*\.?[0-9]+)", t, flags=re.IGNORECASE)
        if m:
            try:
                val = float(m.group(1))
                return max(0.0, min(1.0, val))
            except Exception:
                return None
        return None

    relevance = find_score("Relevance")
    clarity = find_score("Clarity")
    conciseness = find_score("Conciseness")
    creativity = find_score("Creativity")
    overall = find_score("Overall")

    # Explanation: take everything after the line that starts with Explanation
    expl = ""
    m_exp = re.search(r"Explanation\s*:\s*(.*)", t, flags=re.IGNORECASE | re.DOTALL)
    if m_exp:
        expl = m_exp.group(1).strip()
    else:
        parts = t.strip().split("\n\n")
        if parts:
            expl = parts[-1].strip()

    return relevance, clarity, conciseness, creativity, overall, expl

# ---------------- Rotating Gemini Judge Client ----------------
class RotatingGeminiJudge:
    def __init__(self, keys: List[str], model_name: str):
        assert keys, "Provide at least one API key"
        self.keys = keys
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

    def judge_once(self, reference_caption: str, generated_caption: str):
        prompt = PROMPT_TEMPLATE.format(reference_caption=reference_caption.replace('"', '\\"'),
                                       generated_caption=generated_caption.replace('"', '\\"'))
        model = genai.GenerativeModel(self.model_name)
        resp = model.generate_content([prompt])
        if hasattr(resp, "text") and resp.text:
            return resp.text
        elif hasattr(resp, "candidates") and resp.candidates:
            try:
                return resp.candidates[0].content.parts[0].text
            except Exception:
                return str(resp.candidates[0])
        return ""

    def judge_with_rotation(self, reference_caption: str, generated_caption: str, max_attempts:int = 10, backoff_base: float = 1.0):
        attempt = 0
        last_exc = None
        while attempt < max_attempts:
            attempt += 1
            try:
                return self.judge_once(reference_caption, generated_caption)
            except Exception as e:
                last_exc = e
                msg = str(e).lower()
                if any(k in msg for k in ("429", "quota", "rate limit", "quota exceeded", "rate_limit")):
                    print(f"❗ Quota/rate-limit detected on key index {self.key_index}: {e}")
                    self._advance_key()
                    time.sleep(1.0)
                    continue
                print(f"❗ Judge API call failed on key index {self.key_index} (attempt {attempt}/{max_attempts}): {e}")
                self._advance_key()
                sleep_time = backoff_base * (2 ** (attempt - 1))
                print(f"⏳ backing off for {sleep_time:.1f}s before retrying this item")
                time.sleep(min(sleep_time, 60))
                continue

        print(f"❌ Failed to judge after {max_attempts} attempts. Last error: {last_exc}")
        raise last_exc

# ----- Robust JSON mapping helpers & processing -----
def build_map_safe(items: List[dict], key_name: str = "image_id"):
    """
    Build a dict mapping image_id -> caption, but skip and warn on malformed items.
    Returns (map, malformed_count)
    """
    m = {}
    bad = 0
    for i, it in enumerate(items):
        if not isinstance(it, dict):
            print(f"⚠️ Skipping non-object entry at index {i}: {it!r}")
            bad += 1
            continue
        img = it.get(key_name) or it.get("image_id")  # accept either explicit key or fallback
        cap = it.get("caption")
        if img is None:
            # try common typos (e.g., food_id)
            fallback_keys = [k for k in it.keys() if k.endswith("_id")]
            if fallback_keys:
                guessed = it.get(fallback_keys[0])
                if guessed:
                    print(f"⚠️ Found nonstandard id key '{fallback_keys[0]}' -> treating as image_id: {guessed}")
                    img = guessed
            if img is None:
                print(f"⚠️ Skipping item with no image id (index {i}): keys={list(it.keys())}")
                bad += 1
                continue
        if cap is None:
            print(f"⚠️ Item {img} missing 'caption' — treating caption as empty string.")
            cap = ""
        m[str(img)] = str(cap)
    return m, bad

def process_sector_with_rotation(sector_name: str, cfg: dict, client: RotatingGeminiJudge, sleep_after: float = 0.25):
    print(f"\n==== Processing sector: {sector_name} ====")
    ann_path: Path = cfg["annotation"]
    gen_zero_path: Path = cfg["generated_captions_zero"]
    gen_few_path: Path = cfg["generated_captions_few"]

    references = load_json(ann_path)
    gen_zero = load_json(gen_zero_path)
    gen_few = load_json(gen_few_path)

    # build maps safely (handle typos like "food_id")
    ref_map, ref_bad = build_map_safe(references, key_name="image_id")
    gen_zero_map, zero_bad = build_map_safe(gen_zero, key_name="image_id")
    gen_few_map, few_bad = build_map_safe(gen_few, key_name="image_id")

    if ref_bad:
        print(f"⚠️ {ref_bad} malformed/ignored entries in annotation file {ann_path}")
    if zero_bad:
        print(f"⚠️ {zero_bad} malformed/ignored entries in generated zero-shot file {gen_zero_path}")
    if few_bad:
        print(f"⚠️ {few_bad} malformed/ignored entries in generated few-shot file {gen_few_path}")

    # output paths
    out_zero = OUTPUT_ROOT / f"{sector_name}_captions_judged_gemini_zero.json"
    out_few = OUTPUT_ROOT / f"{sector_name}_captions_judged_gemini_few.json"
    out_summary = OUTPUT_ROOT / f"{sector_name}_captions_judged_summary_gemini.json"

    # load existing progress if present
    results_zero, done_zero = ([], set())
    results_few, done_few = ([], set())
    if out_zero.exists():
        try:
            results_zero = load_json(out_zero)
            done_zero = {r["image_id"] for r in results_zero if isinstance(r, dict) and "image_id" in r}
        except Exception:
            results_zero, done_zero = ([], set())
    if out_few.exists():
        try:
            results_few = load_json(out_few)
            done_few = {r["image_id"] for r in results_few if isinstance(r, dict) and "image_id" in r}
        except Exception:
            results_few, done_few = ([], set())

    # order: iterate over union of reference keys and generated keys to avoid KeyError
    image_ids = list(dict.fromkeys(list(ref_map.keys()) + list(gen_zero_map.keys()) + list(gen_few_map.keys())))

    for image_id in image_ids:
        ref_caption = ref_map.get(image_id)
        if ref_caption is None:
            print(f"⚠️ No reference caption found for {image_id} — skipping judging for any generated captions for this image.")
            # If you want to still judge against empty reference, change this behavior.
            continue

        # ZERO-SHOT
        if image_id in gen_zero_map and image_id not in done_zero:
            gen_cap = gen_zero_map[image_id]
            print(f"Judging ZERO for {image_id} (key idx {client.key_index})")
            try:
                raw = client.judge_with_rotation(ref_caption, gen_cap, max_attempts=10)
            except Exception as e:
                print(f"❗ Skipping ZERO for {image_id} due to repeated failures: {e}")
                raw = ""

            rel, cla, con, cre, ovr, expl = parse_judge_response(raw)
            item = {
                "image_id": image_id,
                "reference_caption": ref_caption,
                "generated_caption": gen_cap,
                "raw_judge_output": raw,
                "relevance": clamp_score(rel),
                "clarity": clamp_score(cla),
                "conciseness": clamp_score(con),
                "creativity": clamp_score(cre),
                "overall": clamp_score(ovr),
                "explanation": expl
            }
            results_zero.append(item)
            save_json_atomic(out_zero, results_zero)
            done_zero.add(image_id)
            time.sleep(sleep_after)
        elif image_id not in gen_zero_map:
            print(f"⚠️ Zero-shot caption missing for {image_id}")

        # FEW-SHOT
        if image_id in gen_few_map and image_id not in done_few:
            gen_cap = gen_few_map[image_id]
            print(f"Judging FEW for {image_id} (key idx {client.key_index})")
            try:
                raw = client.judge_with_rotation(ref_caption, gen_cap, max_attempts=10)
            except Exception as e:
                print(f"❗ Skipping FEW for {image_id} due to repeated failures: {e}")
                raw = ""

            rel, cla, con, cre, ovr, expl = parse_judge_response(raw)
            item = {
                "image_id": image_id,
                "reference_caption": ref_caption,
                "generated_caption": gen_cap,
                "raw_judge_output": raw,
                "relevance": clamp_score(rel),
                "clarity": clamp_score(cla),
                "conciseness": clamp_score(con),
                "creativity": clamp_score(cre),
                "overall": clamp_score(ovr),
                "explanation": expl
            }
            results_few.append(item)
            save_json_atomic(out_few, results_few)
            done_few.add(image_id)
            time.sleep(sleep_after)
        elif image_id not in gen_few_map:
            print(f"⚠️ Few-shot caption missing for {image_id}")

    # After processing all images, compute overall aggregates for zero & few
    def aggregate(results: List[dict]) -> Dict:
        if not results:
            return {"n": 0}
        n = len(results)
        sums = {"relevance": 0.0, "clarity": 0.0, "conciseness": 0.0, "creativity": 0.0, "overall": 0.0}
        counts = {k: 0 for k in sums}
        for r in results:
            for k in sums:
                v = r.get(k)
                if v is None:
                    continue
                sums[k] += float(v)
                counts[k] += 1
        means = {}
        for k in sums:
            means[k] = round((sums[k] / counts[k]) if counts[k] > 0 else None, 4)
        means["n"] = n
        return means

    summary = {
        "sector": sector_name,
        "zero_shot": aggregate(results_zero),
        "few_shot": aggregate(results_few)
    }
    save_json_atomic(out_summary, summary)
    print(f"Saved per-image results and summary for sector {sector_name} -> {out_zero}, {out_few}, {out_summary}")

    # Return whether this sector had both generated files fully processed (useful for skip logic)
    fully_done_zero = set(gen_zero_map.keys()).issubset(done_zero) and len(gen_zero_map) > 0
    fully_done_few = set(gen_few_map.keys()).issubset(done_few) and len(gen_few_map) > 0
    return fully_done_zero and fully_done_few

# ---------------- Main ----------------

def main():
    # configure first key before instantiating client (client will also configure)
    genai.configure(api_key=KEY_LIST[0])
    client = RotatingGeminiJudge(KEY_LIST, judge_model)

    # Controls:
    # - STOP_AFTER_FIRST_COMPLETE: if True the entire run stops when a sector is found
    #   where both zero+few generated lists were already fully judged.
    #   (You previously had True - that caused the run to stop after 'culture'.)
    STOP_AFTER_FIRST_COMPLETE = False

    # - SKIP_IF_ALREADY_FULLY_JUDGED: if True and a sector's generated lists are already
    #   fully judged (the function returns True), the script will skip re-processing that sector
    #   and move to the next one.
    SKIP_IF_ALREADY_FULLY_JUDGED = True

    for sector_name, cfg in SECTORS.items():
        try:
            # process sector; function returns whether sector is fully judged
            completed = process_sector_with_rotation(sector_name, cfg, client, sleep_after=0.3)

            if completed and SKIP_IF_ALREADY_FULLY_JUDGED:
                # If the sector was already fully judged (or just completed now),
                # we don't re-run it; we print a clear message and continue.
                print(f"ℹ️ Sector '{sector_name}' is fully judged (zero+few). Continuing to next sector.")
                # If you wanted to STOP the entire run when you see a completed sector,
                # set STOP_AFTER_FIRST_COMPLETE = True. Currently it's False.
                if STOP_AFTER_FIRST_COMPLETE:
                    print(f"✅ STOP_AFTER_FIRST_COMPLETE is True — stopping the whole run as requested.")
                    break
                continue

            # If not completed, we already processed (or re-processed) it above.
            # Continue to next sector in any case.
        except Exception as e:
            print(f"❗ Failed sector {sector_name}: {e}")

if __name__ == "__main__":
    main()

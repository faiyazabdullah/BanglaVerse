import json
import time
from pathlib import Path
from typing import List, Dict, Optional
from comet import download_model, load_from_checkpoint

# ---------------- Paths ----------------
OUTPUT_ROOT = Path(r"...\{model_name}_score")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# ---------------- SECTORS configuration ----------------
SECTORS = {
    "culture": {
        "generated_captions_zero": Path(r"...\{model_name}_csu\culture_csu_results_{model_name}_zero.json"),
        "generated_captions_few": Path(r"...\{model_name}_csu\culture_csu_results_{model_name}_few.json"),
        "annotation": Path(r"...\data\culture\annotations\culture_commonsense_reasoning.json")
    },
    "food": {
        "generated_captions_zero": Path(r"...\{model_name}_csu\food_csu_results_{model_name}_zero.json"),
        "generated_captions_few": Path(r"...\{model_name}_csu\food_csu_results_{model_name}_few.json"),
        "annotation": Path(r"...\data\food\annotations\food_commonsense_reasoning.json")
    },
    "history": {
        "generated_captions_zero": Path(r"...\{model_name}_csu\history_csu_results_{model_name}_zero.json"),
        "generated_captions_few": Path(r"...\{model_name}_csu\history_csu_results_{model_name}_few.json"),
        "annotation": Path(r"...\data\history\annotations\history_commonsense_reasoning.json")
    },
    "media_and_movies": {
        "generated_captions_zero": Path(r"...\{model_name}_csu\media_and_movies_csu_results_{model_name}_zero.json"),
        "generated_captions_few": Path(r"...\{model_name}_csu\media_and_movies_csu_results_{model_name}_few.json"),
        "annotation": Path(r"...\data\media_and_movies\annotations\media_and_movies_commonsense_reasoning.json")
    },
    "national_achievements": {
        "generated_captions_zero": Path(r"...\{model_name}_csu\national_achievements_csu_results_{model_name}_zero.json"),
        "generated_captions_few": Path(r"...\{model_name}_csu\national_achievements_csu_results_{model_name}_few.json"),
        "annotation": Path(r"...\data\national_achievements\annotations\national_achievements_commonsense_reasoning.json")
    },
    "nature": {
        "generated_captions_zero": Path(r"...\{model_name}_csu\nature_csu_results_{model_name}_zero.json"),
        "generated_captions_few": Path(r"...\{model_name}_csu\nature_csu_results_{model_name}_few.json"),
        "annotation": Path(r"...\data\nature\annotations\nature_commonsense_reasoning.json")
    },
    "personalities": {
        "generated_captions_zero": Path(r"...\{model_name}_csu\personalities_csu_results_{model_name}_zero.json"),
        "generated_captions_few": Path(r"...\{model_name}_csu\personalities_csu_results_{model_name}_few.json"),
        "annotation": Path(r"...\data\personalities\annotations\personalities_commonsense_reasoning.json")
    },
    "politics": {
        "generated_captions_zero": Path(r"...\{model_name}_csu\politics_csu_results_{model_name}_zero.json"),
        "generated_captions_few": Path(r"...\{model_name}_csu\politics_csu_results_{model_name}_few.json"),
        "annotation": Path(r"...\data\politics\annotations\politics_commonsense_reasoning.json")
    },
    "sports": {
        "generated_captions_zero": Path(r"...\{model_name}_csu\sports_csu_results_{model_name}_zero.json"),
        "generated_captions_few": Path(r"...\{model_name}_csu\sports_csu_results_{model_name}_few.json"),
        "annotation": Path(r"...\data\sports\annotations\sports_commonsense_reasoning.json")
    }
}

# ---------------- Helpers ----------------
def load_json(path: Path):
    if not path.exists():
        print(f"⚠️ File not found: {path}")
        return []
    with open(path, "r", encoding="utf8") as f:
        return json.load(f)


def save_json_atomic(path: Path, data):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


# ---------------- COMET Model ----------------
print("⏳ Loading COMET model...")
comet_model_path = download_model("Unbabel/wmt22-comet-da")
comet_model = load_from_checkpoint(comet_model_path)
print("✅ COMET model loaded.")


def get_comet_score(reference: str, generated: str) -> Optional[float]:
    try:
        data = [{"src": reference, "mt": generated, "ref": reference}]
        scores = comet_model.predict(data, batch_size=8, gpus=0)
        return float(scores["scores"][0])
    except Exception as e:
        print(f"⚠️ COMET scoring failed: {e}")
        return None


# ---------------- Core Processing ----------------
def process_sector(sector_name: str, cfg: dict, sleep_after: float = 0.0):
    print(f"\n==== Processing sector: {sector_name} ====")

    # Load files
    references = load_json(cfg["annotation"])
    gen_zero = load_json(cfg["generated_captions_zero"])
    gen_few = load_json(cfg["generated_captions_few"])

    # Build maps by image_id
    # Annotation: use "answer" as reference
    ref_map = {item["image_id"]: item["answer"] for item in references}
    # Generated: use "predicted_answer" as generated caption
    gen_zero_map = {item["image_id"]: item["predicted_answer"] for item in gen_zero}
    gen_few_map = {item["image_id"]: item["predicted_answer"] for item in gen_few}

    # Output paths
    out_zero = OUTPUT_ROOT / f"{sector_name}_csu_comet_zero.json"
    out_few = OUTPUT_ROOT / f"{sector_name}_csu_comet_few.json"
    out_summary = OUTPUT_ROOT / f"{sector_name}_csu_comet_summary.json"

    results_zero = []
    results_few = []

    image_ids = list(ref_map.keys())

    for image_id in image_ids:
        ref_caption = ref_map.get(image_id)

        # ZERO-SHOT
        if image_id in gen_zero_map:
            gen_cap = gen_zero_map[image_id]
            comet_score = get_comet_score(ref_caption, gen_cap)
            results_zero.append({
                "image_id": image_id,
                "reference_caption": ref_caption,
                "generated_caption": gen_cap,
                "comet_score": comet_score
            })
            time.sleep(sleep_after)

        # FEW-SHOT
        if image_id in gen_few_map:
            gen_cap = gen_few_map[image_id]
            comet_score = get_comet_score(ref_caption, gen_cap)
            results_few.append({
                "image_id": image_id,
                "reference_caption": ref_caption,
                "generated_caption": gen_cap,
                "comet_score": comet_score
            })
            time.sleep(sleep_after)

    save_json_atomic(out_zero, results_zero)
    save_json_atomic(out_few, results_few)

    # Aggregates
    def aggregate(results: List[dict]) -> Dict:
        if not results:
            return {"n": 0}
        scores = [r["comet_score"] for r in results if r.get("comet_score") is not None]
        return {
            "average_comet_score": round(sum(scores) / len(scores), 4) if scores else None,
            "n": len(scores)
        }

    summary = {
        "sector": sector_name,
        "zero_shot": aggregate(results_zero),
        "few_shot": aggregate(results_few)
    }
    save_json_atomic(out_summary, summary)

    print(f"✅ Saved results for {sector_name} -> {out_zero}, {out_few}, {out_summary}")


# ---------------- Main ----------------
def main():
    for sector_name, cfg in SECTORS.items():
        try:
            process_sector(sector_name, cfg, sleep_after=0.0)
        except Exception as e:
            print(f"❗ Failed sector {sector_name}: {e}")


if __name__ == "__main__":
    main()

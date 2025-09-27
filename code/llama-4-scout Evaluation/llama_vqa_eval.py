import os
import json
import time
import base64
from pathlib import Path
from groq import Groq
from tqdm import tqdm

# ---------------- CONFIG ----------------
API_KEYS = [
    "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",  # Key 1
]
MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

# Process all sectors and 10 examples per sector
N_EXAMPLES = None # set to None to process all images in each sector folder

# Paths
OUTPUT_ROOT = Path(r"...\llama_vqa")
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

IMG_EXTS = {".png", ".jpg", ".jpeg"}

# ---------------- helpers ----------------
def list_images(folder: Path):
    if not folder.exists():
        print(f"⚠️ Image folder not found: {folder}")
        return []
    return sorted([p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS])

def encode_image_to_b64(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def load_vqa_annotations(path: Path):
    if not path.exists():
        print(f"⚠️ Annotation file not found: {path}")
        return []
    with open(path, "r", encoding="utf8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else []

def extract_index_from_answer(answer_text):
    import re
    match = re.search(r'Index\s*:\s*(\d+)', answer_text)
    if match:
        return int(match.group(1))
    return None

def compute_accuracy(preds, gts):
    correct = 0
    total = len(preds)
    for pred_idx, gt_idx in zip(preds, gts):
        if pred_idx == gt_idx:
            correct += 1
    return round(100.0 * correct / total, 2) if total > 0 else 0.0

# ---------------- main flow ----------------
def process_sector(sector_name, sector_cfg, prompt_mode="few", n_examples=None, api_key_index=0):
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

    for item in tqdm(vqa_data, desc=f"VQA ({sector_name})"):
        image_id = item.get("image_id")
        question = item.get("question")
        options = item.get("options")
        answer = item.get("answer")  # Ground truth answer string

        # Find ground truth index
        try:
            gt_index = options.index(answer)
        except Exception:
            gt_index = None

        img_path = images_dir / f"{image_id}.png"
        if not img_path.exists():
            img_path = images_dir / f"{image_id}.jpg"
        if not img_path.exists():
            img_path = images_dir / f"{image_id}.jpeg"
        if not img_path.exists():
            print(f"⚠️ Image not found for {image_id}")
            continue

        b64 = encode_image_to_b64(img_path)
        prompt = PROMPT_FEW_SHOT if prompt_mode == "few" else PROMPT_ZERO_SHOT
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt.format(image_path=img_path, question=question, options=options)},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
                ]
            }
        ]

        while api_key_index < len(API_KEYS):
            try:
                client = Groq(api_key=API_KEYS[api_key_index])
                print(f"Sending {img_path.name} to model with API key {api_key_index + 1}...")
                completion = client.chat.completions.create(
                    model=MODEL,
                    messages=messages,
                    temperature=0.2,
                    max_completion_tokens=128,
                    top_p=1,
                    stream=False
                )
                answer_text = completion.choices[0].message.content.strip()
                break
            except Exception as e:
                print(f"❗ API key {api_key_index + 1} failed: {e}")
                api_key_index += 1
                if api_key_index >= len(API_KEYS):
                    print("❌ All API keys exhausted. Stopping.")
                    return

        pred_index = extract_index_from_answer(answer_text)
        pred_indices.append(pred_index)
        gt_indices.append(gt_index)
        generated.append({
            "image_id": image_id,
            "question": question,
            "options": options,
            "ground_truth_index": gt_index,
            "ground_truth_answer": answer,
            "predicted_index": pred_index,
            "answer_text": answer_text
        })
        time.sleep(0.5)

    accuracy = compute_accuracy(pred_indices, gt_indices)
    metrics = {"Accuracy (%)": accuracy, "n_examples": len(generated)}

    out_file = OUTPUT_ROOT / f"{sector_name}_vqa_llama_{prompt_mode}.json"
    with open(out_file, "w", encoding="utf8") as f:
        json.dump(generated, f, ensure_ascii=False, indent=2)
    print("Saved results to:", out_file)

    metrics_out = OUTPUT_ROOT / f"{sector_name}_vqa_metrics_llama_{prompt_mode}.json"
    with open(metrics_out, "w", encoding="utf8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print("Saved metrics to:", metrics_out)

def main():
    for prompt_mode in ["zero", "few"]:
        for sector_name, sector_cfg in SECTORS.items():
            try:
                process_sector(sector_name, sector_cfg, prompt_mode=prompt_mode, n_examples=N_EXAMPLES)
            except Exception as e:
                print(f"❗ Error processing sector {sector_name} ({prompt_mode}-shot): {e}")

if __name__ == "__main__":
    main()
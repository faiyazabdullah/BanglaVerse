import os
import json
import time
from pathlib import Path
from tqdm import tqdm
import google.generativeai as genai

API_KEY = "AIzaSyCpoEuW063lZTYYVWvWbcassLMA4SQzHd8"
MODEL = "gemini-2.5-flash"
IMG_EXTS = {".png", ".jpg", ".jpeg"}

OUTPUT_ROOT = Path(r"F:\Labib\Labib Folder\Labib\Research\BanglaVerse Experiments\BanglaVerse\gemini_vqa")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

SECTORS = {
    "culture": {
        "images": Path(r"F:\Labib\Labib Folder\Labib\Research\BanglaVerse Experiments\BanglaVerse\data\culture\images"),
        "annotation": Path(r"F:\Labib\Labib Folder\Labib\Research\BanglaVerse Experiments\BanglaVerse\data\culture\annotations\culture_qa_pairs.json")
    },
    "food": {
        "images": Path(r"F:\Labib\Labib Folder\Labib\Research\BanglaVerse Experiments\BanglaVerse\data\food\images"),
        "annotation": Path(r"F:\Labib\Labib Folder\Labib\Research\BanglaVerse Experiments\BanglaVerse\data\food\annotations\food_qa_pairs.json")
    },
    "history": {
        "images": Path(r"F:\Labib\Labib Folder\Labib\Research\BanglaVerse Experiments\BanglaVerse\data\history\images"),
        "annotation": Path(r"F:\Labib\Labib Folder\Labib\Research\BanglaVerse Experiments\BanglaVerse\data\history\annotations\history_qa_pairs.json")
    },
    "media_and_movies": {
        "images": Path(r"F:\Labib\Labib Folder\Labib\Research\BanglaVerse Experiments\BanglaVerse\data\media_and_movies\images"),
        "annotation": Path(r"F:\Labib\Labib Folder\Labib\Research\BanglaVerse Experiments\BanglaVerse\data\media_and_movies\annotations\media_and_movies_qa_pairs.json")
    },
    "national_achievements": {
        "images": Path(r"F:\Labib\Labib Folder\Labib\Research\BanglaVerse Experiments\BanglaVerse\data\national_achievements\images"),
        "annotation": Path(r"F:\Labib\Labib Folder\Labib\Research\BanglaVerse Experiments\BanglaVerse\data\national_achievements\annotations\national_achievements_qa_pairs.json")
    },
    "nature": {
        "images": Path(r"F:\Labib\Labib Folder\Labib\Research\BanglaVerse Experiments\BanglaVerse\data\nature\images"),
        "annotation": Path(r"F:\Labib\Labib Folder\Labib\Research\BanglaVerse Experiments\BanglaVerse\data\nature\annotations\nature_qa_pairs.json")
    },
    "personalities": {
        "images": Path(r"F:\Labib\Labib Folder\Labib\Research\BanglaVerse Experiments\BanglaVerse\data\personalities\images"),
        "annotation": Path(r"F:\Labib\Labib Folder\Labib\Research\BanglaVerse Experiments\BanglaVerse\data\personalities\annotations\personalities_qa_pairs.json")
    },
    "politics": {
        "images": Path(r"F:\Labib\Labib Folder\Labib\Research\BanglaVerse Experiments\BanglaVerse\data\politics\images"),
        "annotation": Path(r"F:\Labib\Labib Folder\Labib\Research\BanglaVerse Experiments\BanglaVerse\data\politics\annotations\politics_qa_pairs.json")
    },
    "sports": {
        "images": Path(r"F:\Labib\Labib Folder\Labib\Research\BanglaVerse Experiments\BanglaVerse\data\sports\images"),
        "annotation": Path(r"F:\Labib\Labib Folder\Labib\Research\BanglaVerse Experiments\BanglaVerse\data\sports\annotations\sports_qa_pairs.json")
    }
}

PROMPT_ZERO_SHOT = (
    "You are an assistant that answers visual multiple-choice questions in Bangla. "
    "Look at the given {image_path} and the {question}. Choose the most accurate answer with list index from the provided list {options}\n"
    "Always return the answer in this format:\n"
    "Index: <option_index>, Answer: <option_text_in_Bangla>"
)

PROMPT_FEW_SHOT = (
    "You are an assistant that answers visual multiple-choice questions in Bangla. "
    "Look at the given {image_path} and the {question}. Choose the most accurate answer from the provided list {options}.\n"
    "Always return the answer in this format:\n"
    "Index: <option_index>, Answer: <option_text_in_Bangla>\n"
    "Examples:\n"
    "Image: ./dataset/culture/images/culture_003.png\n"
    "Question: \"ছবিতে কোন উৎসব পালিত হচ্ছে?\"\n"
    "Options: [\"পহেলা বৈশাখ\", \"ঈদ\",  \"নববর্ষ\",  \"নবন্ন\"]\n"
    "Answer: Index: 0, Answer: \"পহেলা বৈশাখ\"\n"
    "Image: ./dataset/history/images/history_002.png\n"
    "Question: \"এই ঘটনার পর বাংলাদেশে কী পরিবর্তন ঘটে?\"\n"
    "Options: [\"সংবিধান প্রণয়ন\", \"স্বাধীনতা অর্জন\", \"সেনা শাসনের শুরু\", \"গণভোট\"]\n"
    "Answer: Index: 1, Answer: \"স্বাধীনতা অর্জন\"\n"
    "Image: ./dataset/sports/images/sports_001.png\n"
    "Question: \"এই ক্রিকেটারটি কে?\"\n"
    "Options: [\"মাশরাফি বিন মোর্ত্তজা\", \"সাকিব আল হাসান\", \"মুশফিকুর রহিম\", \"তামিম ইকবাল\"]\n"
    "Answer: Index: 1, Answer: \"সাকিব আল হাসান\"\n"
    "Now, answer for the given image."
)

class GeminiVQA:
    def __init__(self, api_key, model):
        genai.configure(api_key=api_key)
        self.model = model

    def ask(self, image_path, question, options, prompt_mode="few", max_retries=3, sleep_between=1.0):
        # Only few-shot prompt is used
        prompt = PROMPT_FEW_SHOT + f"\nUser-input\n{str(image_path)}\n\"{question}\"\n{json.dumps(options, ensure_ascii=False)}"
        for attempt in range(max_retries):
            try:
                img = genai.upload_file(str(image_path))
                model = genai.GenerativeModel(self.model)
                resp = model.generate_content([prompt, img])
                if hasattr(resp, "text") and resp.text:
                    return resp.text.strip()
                elif hasattr(resp, "candidates") and resp.candidates:
                    cand_text = resp.candidates[0].content.parts[0].text
                    return cand_text.strip()
                else:
                    return "⚠️ Empty response"
            except Exception as e:
                if "429" in str(e) or "quota" in str(e).lower():
                    print(f"❗ Gemini quota exceeded for {image_path.name}: {e}")
                    return "Caption not available due to Gemini API quota limits."
                print(f"❗ Gemini call failed for {image_path.name} (attempt {attempt+1}/{max_retries}): {e}")
                time.sleep(sleep_between * (attempt+1))
        return "❌ Failed to answer"

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

def extract_index_from_answer(answer_text):
    # Extracts the index from "Index: <option_index>, Answer: ..."
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

def process_sector(sector_name, sector_cfg, prompt_mode="few", n_examples=10):
    print(f"\n==== Processing sector: {sector_name} (prompt_mode={prompt_mode}) ====")
    images_dir = sector_cfg["images"]
    annotation_file = sector_cfg["annotation"]

    vqa_data = load_vqa_annotations(annotation_file)
    vqa_data = vqa_data[:n_examples]  # Only process first 10 examples

    if not vqa_data:
        return

    vqa_model = GeminiVQA(API_KEY, MODEL)
    results, pred_indices, gt_indices = [], [], []

    for item in tqdm(vqa_data, desc=f"VQA ({sector_name})"):
        image_id = item.get("image_id")
        question = item.get("question")
        options = item.get("options")
        answer = item.get("answer")  # The ground truth answer string

        # Find ground truth index by matching answer string to options
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

        answer_text = vqa_model.ask(img_path, question, options, prompt_mode=prompt_mode)
        pred_index = extract_index_from_answer(answer_text)
        pred_indices.append(pred_index)
        gt_indices.append(gt_index)
        results.append({
            "image_id": image_id,
            "question": question,
            "options": options,
            "ground_truth_index": gt_index,
            "ground_truth_answer": answer,
            "predicted_index": pred_index,
            "answer_text": answer_text
        })
        time.sleep(20)  # Increased delay to avoid 429 errors

    accuracy = compute_accuracy(pred_indices, gt_indices)
    metrics = {"Accuracy (%)": accuracy, "n_examples": len(results)}

    out_file = OUTPUT_ROOT / f"{sector_name}_vqa_gemini_{prompt_mode}.json"
    with open(out_file, "w", encoding="utf8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    metrics_out = OUTPUT_ROOT / f"{sector_name}_vqa_metrics_gemini_{prompt_mode}.json"
    with open(metrics_out, "w", encoding="utf8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

def main():
    # for prompt_mode in ["zero", "few"]:
    #     for sector_name, cfg in SECTORS.items():
    #         try:
    #             process_sector(sector_name, cfg, prompt_mode=prompt_mode, n_examples=10)
    #         except Exception as e:
    #             print(f"❗ Error processing sector {sector_name} ({prompt_mode}-shot): {e}")
    # Only run for few-shot
    for sector_name, cfg in SECTORS.items():
        try:
            process_sector(sector_name, cfg, prompt_mode="few", n_examples=10)
        except Exception as e:
            print(f"❗ Error processing sector {sector_name} (few-shot): {e}")

if __name__ == "__main__":
    main()
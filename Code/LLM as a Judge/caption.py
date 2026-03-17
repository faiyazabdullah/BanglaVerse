import json
import random
from pathlib import Path
from typing import Dict, List, Tuple
import requests
from bert_score import score as bert_score
import torch

# Sectors configuration
SECTORS = {
    # Corresponding Sectors
}

# Paths
GENERATED_CAPTIONS_DIR = Path(r"...") # Your generated Caption directory
LLM_JUDGE_PROMPT_PATH = Path(r"/mnt/wwn-0x500a0751e14d807e-part2/Labib/Labib Folder/Labib/Research/BanglaVerse_2/BanglaVerseV2 Experiment/prompts/LLM-as-a-Judge.txt")

# Ollama configuration
OLLAMA_API_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "gpt-oss:20b"


def load_json(file_path: Path) -> List[Dict]:
    """Load JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: Dict, file_path: Path):
    """Save dictionary to JSON file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def get_average_caption_length(annotations: List[Dict]) -> float:
    """Calculate average caption length in characters."""
    total_length = sum(len(item['caption']) for item in annotations)
    return total_length / len(annotations)


def get_valid_samples(
    generated_captions: List[Dict],
    ground_truth_annotations: List[Dict],
    avg_length: float
) -> List[Tuple[Dict, Dict]]:
    """
    Get all samples where generated caption length < average ground truth length and not empty.
    
    Returns list of tuples: (generated_item, ground_truth_item)
    """
    # Create a mapping of image_id to ground truth
    gt_map = {item['image_id']: item for item in ground_truth_annotations}
    
    # Filter valid samples (generated length < avg length and not empty)
    valid_samples = []
    for gen_item in generated_captions:
        image_id = gen_item['image_id']
        if image_id in gt_map:
            gen_caption = gen_item['caption']
            gt_item = gt_map[image_id]
            
            # Only include if caption is not empty and length < average
            if gen_caption and len(gen_caption) < avg_length:
                valid_samples.append((gen_item, gt_item))
    
    return valid_samples


def call_ollama_llm(reference_caption: str, generated_caption: str, prompt_template: str) -> Dict:
    """Call Ollama API for LLM-as-a-Judge evaluation."""
    # Format prompt
    prompt = prompt_template.format(
        reference_caption=reference_caption,
        generated_caption=generated_caption
    )
    
    # Prepare request
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False
    }
    
    try:
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=120)
        response.raise_for_status()
        result = response.json()
        
        # Parse the response
        response_text = result.get('response', '')
        return parse_llm_response(response_text)
        
    except Exception as e:
        print(f"Error calling Ollama API: {e}")
        return {
            "Relevance": 0.0,
            "Clarity": 0.0,
            "Conciseness": 0.0,
            "Creativity": 0.0,
            "Overall": 0.0,
            "Explanation": f"Error: {str(e)}"
        }


def parse_llm_response(response_text: str) -> Dict:
    """Parse LLM response to extract scores."""
    scores = {
        "Relevance": 0.0,
        "Clarity": 0.0,
        "Conciseness": 0.0,
        "Creativity": 0.0,
        "Overall": 0.0,
        "Explanation": ""
    }
    
    lines = response_text.strip().split('\n')
    explanation_started = False
    explanation_lines = []
    
    for line in lines:
        line = line.strip()
        if ':' in line and not explanation_started:
            key, value = line.split(':', 1)
            key = key.strip()
            value = value.strip()
            
            if key in scores and key != "Explanation":
                try:
                    scores[key] = float(value)
                except ValueError:
                    pass
            elif key == "Explanation":
                explanation_started = True
                if value:
                    explanation_lines.append(value)
        elif explanation_started:
            explanation_lines.append(line)
    
    scores["Explanation"] = ' '.join(explanation_lines)
    return scores


def calculate_llm_judge_scores(
    samples: List[Tuple[Dict, Dict]],
    prompt_template: str
) -> List[Dict]:
    """Calculate LLM-as-a-Judge scores for samples."""
    results = []
    
    for idx, (gen_item, gt_item) in enumerate(samples, 1):
        print(f"  Processing LLM Judge sample {idx}/{len(samples)}: {gen_item['image_id']}")
        
        scores = call_ollama_llm(
            reference_caption=gt_item['caption'],
            generated_caption=gen_item['caption'],
            prompt_template=prompt_template
        )
        
        result = {
            "image_id": gen_item['image_id'],
            "generated_caption": gen_item['caption'],
            "reference_caption": gt_item['caption'],
            "llm_judge_scores": scores
        }
        results.append(result)
    
    return results


def calculate_bert_scores(samples: List[Tuple[Dict, Dict]]) -> List[Dict]:
    """Calculate BERTScore for samples."""
    results = []
    
    # Prepare batch data
    generated_captions = [gen_item['caption'] for gen_item, _ in samples]
    reference_captions = [gt_item['caption'] for _, gt_item in samples]
    
    print(f"  Calculating BERTScore for {len(samples)} samples...")
    
    try:
        # Calculate BERTScore
        P, R, F1 = bert_score(
            generated_captions,
            reference_captions,
            lang='bn',
            verbose=True,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        # Convert to list
        P_list = P.tolist()
        R_list = R.tolist()
        F1_list = F1.tolist()
        
        # Create results
        for idx, (gen_item, gt_item) in enumerate(samples):
            result = {
                "image_id": gen_item['image_id'],
                "generated_caption": gen_item['caption'],
                "reference_caption": gt_item['caption'],
                "bert_score": {
                    "precision": P_list[idx],
                    "recall": R_list[idx],
                    "f1": F1_list[idx]
                }
            }
            results.append(result)
            
    except Exception as e:
        print(f"Error calculating BERTScore: {e}")
        # Return empty scores
        for gen_item, gt_item in samples:
            result = {
                "image_id": gen_item['image_id'],
                "generated_caption": gen_item['caption'],
                "reference_caption": gt_item['caption'],
                "bert_score": {
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": 0.0,
                    "error": str(e)
                }
            }
            results.append(result)
    
    return results


def process_sector(sector_name: str, shot_type: str):
    """Process a sector and calculate LLM judge and BERTScore metrics."""
    print(f"\n{'='*80}")
    print(f"Processing: {sector_name} - {shot_type}")
    print(f"{'='*80}")
    
    # Load files
    caption_file = GENERATED_CAPTIONS_DIR / f"{sector_name}_captions_ollama_{shot_type}.json"
    metrics_file = GENERATED_CAPTIONS_DIR / f"{sector_name}_metrics_ollama_{shot_type}.json"
    annotation_file = SECTORS[sector_name]["annotation"]
    
    if not caption_file.exists():
        print(f"Caption file not found: {caption_file}")
        return
    
    if not metrics_file.exists():
        print(f"Metrics file not found: {metrics_file}")
        return
    
    if not annotation_file.exists():
        print(f"Annotation file not found: {annotation_file}")
        return
    
    # Load data
    generated_captions = load_json(caption_file)
    metrics = load_json(metrics_file)
    ground_truth_annotations = load_json(annotation_file)
    
    # Calculate average caption length
    avg_length = get_average_caption_length(ground_truth_annotations)
    print(f"Average ground truth caption length: {avg_length:.2f} characters")
    
    # Get valid samples (all samples where length < avg and not empty)
    samples = get_valid_samples(generated_captions, ground_truth_annotations, avg_length)
    print(f"Processing {len(samples)} valid samples (length < avg, not empty)")
    
    if len(samples) == 0:
        print("No valid samples found. Skipping.")
        return
    
    # Load LLM Judge prompt
    with open(LLM_JUDGE_PROMPT_PATH, 'r', encoding='utf-8') as f:
        llm_prompt_template = f.read().strip().strip('"""')
    
    # Step 1: Calculate LLM Judge scores
    print("\nStep 1: Calculating LLM-as-a-Judge scores...")
    llm_results = calculate_llm_judge_scores(samples, llm_prompt_template)
    
    # Step 2: Calculate BERTScore
    print("\nStep 2: Calculating BERTScore...")
    bert_results = calculate_bert_scores(samples)
    
    # Combine results
    combined_results = []
    for llm_res, bert_res in zip(llm_results, bert_results):
        combined = {
            "image_id": llm_res['image_id'],
            "generated_caption": llm_res['generated_caption'],
            "reference_caption": llm_res['reference_caption'],
            "llm_judge_scores": llm_res['llm_judge_scores'],
            "bert_score": bert_res['bert_score']
        }
        combined_results.append(combined)
    
    # Calculate average scores
    avg_llm_scores = {
        "Relevance": sum(r['llm_judge_scores']['Relevance'] for r in combined_results) / len(combined_results),
        "Clarity": sum(r['llm_judge_scores']['Clarity'] for r in combined_results) / len(combined_results),
        "Conciseness": sum(r['llm_judge_scores']['Conciseness'] for r in combined_results) / len(combined_results),
        "Creativity": sum(r['llm_judge_scores']['Creativity'] for r in combined_results) / len(combined_results),
        "Overall": sum(r['llm_judge_scores']['Overall'] for r in combined_results) / len(combined_results)
    }
    
    avg_bert_score = {
        "precision": sum(r['bert_score']['precision'] for r in combined_results) / len(combined_results),
        "recall": sum(r['bert_score']['recall'] for r in combined_results) / len(combined_results),
        "f1": sum(r['bert_score']['f1'] for r in combined_results) / len(combined_results)
    }
    
    # Update metrics file
    metrics['llm_judge_avg'] = avg_llm_scores
    metrics['bert_score_random_10_avg'] = avg_bert_score
    metrics['llm_judge_detailed_results'] = combined_results
    
    # Save updated metrics
    save_json(metrics, metrics_file)
    print(f"\nMetrics updated and saved to: {metrics_file}")
    
    # Print summary
    print("\n" + "="*80)
    print("Summary:")
    print(f"LLM Judge Average Scores:")
    for key, value in avg_llm_scores.items():
        print(f"  {key}: {value:.4f}")
    print(f"\nBERTScore Average:")
    for key, value in avg_bert_score.items():
        print(f"  {key}: {value:.4f}")
    print("="*80)


def main():
    """Main function to process all sectors."""
    random.seed(42)  # For reproducibility
    
    sectors = list(SECTORS.keys())
    shot_types = ["few", "zero"]
    
    print("Starting evaluation process...")
    print(f"Total sectors: {len(sectors)}")
    print(f"Shot types: {shot_types}")
    
    for sector in sectors:
        for shot_type in shot_types:
            try:
                process_sector(sector, shot_type)
            except Exception as e:
                print(f"\nError processing {sector} - {shot_type}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    print("\n" + "="*80)
    print("All processing completed!")
    print("="*80)


if __name__ == "__main__":
    main()

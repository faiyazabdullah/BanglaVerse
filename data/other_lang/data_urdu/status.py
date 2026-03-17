#!/usr/bin/env python3
"""
Quick Status Checker - Shows what's completed and what's next
"""

import json
from pathlib import Path

def check_status():
    """Check current translation status"""
    target_dir = "/mnt/wwn-0x500a0751e14d807e-part2/Labib/Labib Folder/Labib/Research/BanglaVerse Experiments/BanglaVerse_V2/data_others/data_urdu"
    checkpoint_file = Path(target_dir) / "translation_checkpoint.json"
    
    categories = ['culture', 'food', 'history', 'media_and_movies', 
                  'national_achievements', 'nature', 'personalities', 'politics', 'sports']
    
    file_types = ['captions.json', 'qa_pairs.json', 'commonsense_reasoning.json']
    
    # Load checkpoint
    checkpoint = {}
    if checkpoint_file.exists():
        try:
            with open(checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
        except:
            pass
    
    completed_files = checkpoint.get("completed_files", [])
    current_category = checkpoint.get("current_category")
    current_file = checkpoint.get("current_file")
    current_index = checkpoint.get("current_index", 0)
    
    print("🔄 Urdu Translation Status")
    print("=" * 50)
    
    if current_category and current_file:
        print(f"📍 Currently processing: {current_category}_{current_file}")
        print(f"📊 Progress: Item {current_index}")
        print()
    
    print("✅ Completed Files:")
    for completed in completed_files:
        print(f"  ✓ {completed}")
    
    print(f"\n📈 Total completed: {len(completed_files)}/27 files")
    
    # Show what's next
    all_files = []
    for cat in categories:
        for file_type in file_types:
            file_key = f"{cat}_{cat}_{file_type}"
            all_files.append((cat, f"{cat}_{file_type}", file_key))
    
    print("\n⏳ Remaining Files:")
    count = 0
    for cat, filename, file_key in all_files:
        if file_key not in completed_files:
            if count < 5:  # Show next 5
                print(f"  ⏳ {file_key}")
            count += 1
    
    if count > 5:
        print(f"  ... and {count - 5} more files")
    
    completion = len(completed_files) / 27 * 100
    print(f"\n🎯 Overall Progress: {completion:.1f}%")

if __name__ == "__main__":
    check_status()
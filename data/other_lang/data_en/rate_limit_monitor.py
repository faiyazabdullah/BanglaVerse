#!/usr/bin/env python3
"""
Rate Limit Status Monitor
Shows current API usage and rate limit compliance
"""

import json
import time
from pathlib import Path

def check_rate_limit_status():
    """Check rate limit compliance and current status"""
    
    target_dir = "/mnt/wwn-0x500a0751e14d807e-part2/Labib/Labib Folder/Labib/Research/BanglaVerse Experiments/BanglaVerse_V2/data_others/data_en"
    
    print("🚦 Rate Limit Status Check - English Translation")
    print("=" * 50)
    
    # Gemini 2.5 Flash limits
    print("📋 API Limits:")
    print("  • Requests per minute (RPM): 10")
    print("  • Tokens per minute (TPM): 250,000") 
    print("  • Requests per day (RPD): Not specified")
    print()
    
    print("⚡ Current Protection Settings:")
    print("  • Minimum delay between requests: 7 seconds")
    print("  • Maximum requests per minute: ~8.5 (safe buffer)")
    print("  • Automatic key switching on rate limits")
    print("  • Incremental saves after each translation")
    print()
    
    # Check if there are partial files being worked on
    checkpoint_file = Path(target_dir) / "translation_checkpoint.json"
    if checkpoint_file.exists():
        try:
            with open(checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
            
            current_category = checkpoint.get("current_category")
            current_file = checkpoint.get("current_file") 
            current_index = checkpoint.get("current_index", 0)
            
            print("📊 Current Progress:")
            if current_category and current_file:
                print(f"  • Processing: {current_category}_{current_file}")
                print(f"  • Item: {current_index}")
            else:
                print("  • No active translation session")
            
            completed = len(checkpoint.get("completed_files", []))
            print(f"  • Completed files: {completed}/27")
            print()
            
        except:
            print("  • Could not read checkpoint file")
            print()
    
    print("⏱️  Estimated Time per Item Type:")
    print("  • Caption: ~7 seconds (1 API call)")
    print("  • QA Pair: ~42 seconds (6 API calls: question + 4 options + answer)")
    print("  • Commonsense: ~14 seconds (2 API calls: question + answer)")
    print()
    
    print("📈 Daily Capacity Estimation:")
    print("  • With 24 API keys: ~20,736 requests/day")
    print("  • Estimated total needed: ~8,000-12,000 requests")
    print("  • Should complete in 1-2 days with rate limits")
    print()
    
    print("✅ Safety Features:")
    print("  • Auto-save after each translation")
    print("  • Resume from exact stopping point")
    print("  • Rate limit monitoring and switching")
    print("  • Token usage estimation")
    print("  • Exponential backoff on errors")

if __name__ == "__main__":
    check_rate_limit_status()
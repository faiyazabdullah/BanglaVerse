# BanglaVerse Urdu Translation

This directory contains the Bengali to Urdu translation system for the BanglaVerse dataset.

## Structure

```
data_urdu/
├── data_urdu.py                    # Main translation script
├── rate_limit_monitor.py           # Rate limit monitoring utility
├── status.py                       # Translation progress checker
├── requirements_translation.txt    # Python dependencies
├── translation_checkpoint.json     # Progress tracking (auto-generated)
├── translation_log.txt            # Translation logs (auto-generated)
└── [category]/                    # Translated data by category
    ├── annotations/
    │   ├── [category]_captions.json
    │   ├── [category]_qa_pairs.json
    │   └── [category]_commonsense_reasoning.json
    └── images/                    # Symlink to original images
```

## Usage

### 1. Install Requirements
```bash
pip install -r requirements_translation.txt
```

### 2. Run Translation
```bash
python data_urdu.py
```

### 3. Monitor Progress
```bash
# Check current status
python status.py

# Check rate limit compliance
python rate_limit_monitor.py
```

## Features

- **Rate Limit Protection**: 7-second delays between requests to stay within Gemini API limits
- **Automatic Key Switching**: Cycles through multiple API keys when limits are hit
- **Resume Capability**: Automatically resumes from where it left off using checkpoints
- **Incremental Saving**: Saves progress after each translation to prevent data loss
- **Cultural Context Preservation**: Maintains Bengali cultural references in Urdu translations

## Categories

The system translates 9 categories with 3 file types each (27 total files):

- culture
- food  
- history
- media_and_movies
- national_achievements
- nature
- personalities
- politics
- sports

## Translation Types

- **Captions**: Image descriptions (1 API call per item)
- **QA Pairs**: Questions with multiple choice answers (6 API calls per item)
- **Commonsense Reasoning**: Question-answer pairs (2 API calls per item)

## Rate Limits

- **Gemini 2.5 Flash Lite**: 10 RPM, 250K TPM
- **Protection**: 7-second minimum delays, automatic key rotation
- **Estimated Time**: 1-2 days for complete dataset translation

## Files Generated

- `translation_checkpoint.json`: Tracks completion status and current progress
- `translation_log.txt`: Detailed logs of all translation operations
- Category folders with translated JSON files matching original structure
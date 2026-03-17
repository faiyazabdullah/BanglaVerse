# Bangla to Hindi Translation System

This system translates all Bangla annotations from the BanglaVerse dataset to Hindi using Google's Gemini API with automatic key switching.

## Files Overview

1. **`data_hindi.py`** - Main translation script
2. **`test_setup.py`** - Setup validation script
3. **`progress_tracker.py`** - Progress monitoring script
4. **`requirements_translation.txt`** - Dependencies

## Features

- **Automatic Key Switching**: Cycles through 24 Gemini API keys when rate limits are hit
- **Progress Tracking**: Logs all translation progress and errors
- **Robust Error Handling**: Continues translation even if some items fail
- **Exact Structure Preservation**: Maintains the same JSON structure as original files
- **Comprehensive Coverage**: Handles all categories and annotation types

## Translation Coverage

### Categories:
- culture
- food  
- history
- media_and_movies
- national_achievements
- nature
- personalities
- politics
- sports

### Annotation Types:
- **Captions** (`*_captions.json`) - Image captions
- **QA Pairs** (`*_qa_pairs.json`) - Questions, options, and answers
- **Commonsense Reasoning** (`*_commonsense_reasoning.json`) - Questions and answers

## Usage Instructions

### Step 1: Test Setup
```bash
# Activate virtual environment
source banglaverseV2/bin/activate

# Test the setup
cd data_others/data_hindi
python test_setup.py
```

### Step 2: Start Translation
```bash
# Run the main translation (this will take several hours)
python data_hindi.py
```

### Step 3: Monitor Progress
```bash
# Check progress in another terminal
python progress_tracker.py
```

## Expected Output Structure

The translated files will be saved in:
```
data_others/data_hindi/
├── culture/
│   ├── annotations/
│   │   ├── culture_captions.json
│   │   ├── culture_qa_pairs.json
│   │   └── culture_commonsense_reasoning.json
│   └── images/ -> (symlink to original images)
├── food/
│   └── ...
└── ... (all 9 categories)
```

## Key Features

### Automatic Key Management
- 24 Gemini API keys with automatic rotation
- Handles rate limits gracefully
- Continues from where it left off

### Translation Quality
- Context-aware translation prompts
- Preserves cultural meaning
- Maintains technical terminology

### Error Recovery
- Logs all errors to `translation_log.txt`
- Skips problematic items and continues
- Detailed progress tracking

## Monitoring

### Log File: `translation_log.txt`
- Translation progress
- API key switches
- Error messages
- Completion status

### Progress Tracking
Run `python progress_tracker.py` anytime to see:
- Files completed per category
- Overall completion percentage
- Remaining work

## Estimated Time

With 24 API keys and automatic switching:
- **Small categories** (culture, nature): ~30-60 minutes each
- **Large categories** (qa_pairs): ~2-4 hours each  
- **Total estimated time**: ~8-12 hours for all categories

## Troubleshooting

### If translation stops:
1. Check `translation_log.txt` for errors
2. Run `python progress_tracker.py` to see what's completed
3. Restart with `python data_hindi.py` (it will skip completed files)

### If API keys are exhausted:
- The system will cycle back to the first key
- Add delays between requests in the script if needed

### File Permission Issues:
- Ensure write permissions to `data_others/data_hindi/`
- Check disk space availability

## Quality Assurance

The system includes multiple quality checks:
- Validates JSON structure after translation
- Logs empty or failed translations
- Preserves original text if translation fails
- Creates backups through logging

## File Size Estimates

Expected output sizes:
- **Captions**: ~50-200 items per category
- **QA Pairs**: ~100-500 items per category  
- **Commonsense**: ~50-150 items per category
- **Total**: ~5,000-8,000 translated items across all categories
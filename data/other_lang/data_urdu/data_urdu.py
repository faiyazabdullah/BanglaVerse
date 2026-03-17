#!/usr/bin/env python3
"""
Bangla to Urdu Translation System for BanglaVerse Dataset
Translates all annotation files from Bangla to Urdu using Gemini API with automatic key switching
"""

import json
import os
import time
import google.generativeai as genai
from pathlib import Path
import traceback
from typing import Dict, List, Any
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('translation_log.txt'),
        logging.StreamHandler()
    ]
)

# Gemini API Configuration
KEY_LIST = [
    "AIzaSyBR_78vU5ADRHjx9l8P3aYvAdca-tMHWRA",
    "AIzaSyAIcqatH5JCbXRpOy2Lfjvahign9t5SgjA",
    "AIzaSyD6IiDvyhVjT-LEFJhmVKujKRzLCTzv6Uw",
    "AIzaSyAZxTLUF7qSEE05IGMnTYz6wqYEze1grZ8",
    "AIzaSyANNb9t69WGzB7pTNwHCbgnBk5GYwYi_Z4",
    "AIzaSyBQjQq5krnZhCoAiIo0C6oF9LZagljdByw",
    "AIzaSyABGhNnAfyJua0DKOz4P17Gvyiqu7SFXqw",
    "AIzaSyBN2JvRAzx-waYpfLO-AdC-HDBSXJFvXJ8",
    "AIzaSyDfh7ulSMNVEpZ6Cr0ZIfaZedDtMHJNf0E",
    "AIzaSyD5fLaCbydNxSZJYxYI8xiJuKwVS2Y0aaI",
    "AIzaSyCC-ry-WKMWDaGVcRwYN_FgcUY5sGKsT1s",
    "AIzaSyATQl6OeSpm5kxe2a0TyZ33jAsw__Jqq6M",
    "AIzaSyBmTEInkG9NJlPVh2NVmb99nu_RPUHweus",
    "AIzaSyAEsZUHrrKYKKVysIin6Y30Pycuv9baGAM",
    "AIzaSyBEZ0Vo7A6CCHqX0cAkK-YDuuLbt4dTI5o",
    "AIzaSyACCt9uU3uaIZbsyqZhDzQtRZJkMCcRBPc",
    "AIzaSyCKD8jtDxQswCSkvXDnXs4M_Q6x9Eb9Wu4",
    "AIzaSyAAS1P78GIrbiyzHxYP6v2M4-J-qIwPA8M",
    "AIzaSyDQJO3ux83XfokautqLztl9VMq35sb1bYg",
    "AIzaSyCtLJiYesWR-BNBBodR8n9Fmxo2-ML4qfw",
    "AIzaSyAYEmU0kCQOJfmFUN-qpItdibXeBJlEX9Q",
    "AIzaSyBGAmArN4b3cyaTF59LpTywG5DW9enGnfs",
    "AIzaSyA_Wt7szmlD1wG52qRZxGhGsgR5FNWPnc4",
    "AIzaSyC2gSbOqB8cqKgl8FmbeAtxGb4YxVfESgg",
    "AIzaSyBWQL2D8X0wbW50j2JPKQNTHPM5V7KJRXE",
]

MODEL = "gemini-2.5-flash-lite"

class GeminiTranslator:
    def __init__(self):
        self.current_key_index = 0
        self.model = None
        self.setup_current_key()
        
        # Rate limit tracking
        self.requests_count = 0
        self.tokens_used = 0
        self.last_request_time = 0
        self.daily_requests = 0
        
        # Translation prompts for different types
        self.translation_prompts = {
            'caption': """Translate the following Bengali caption to Urdu. Keep the meaning exactly the same and maintain the cultural context. Only return the Urdu translation, nothing else.

Bengali: {text}
Urdu:""",
            
            'question': """Translate the following Bengali question to Urdu. Keep the meaning exactly the same and maintain the cultural context. Only return the Urdu translation, nothing else.

Bengali: {text}
Urdu:""",
            
            'answer': """Translate the following Bengali answer to Urdu. Keep the meaning exactly the same and maintain the cultural context. Only return the Urdu translation, nothing else.

Bengali: {text}
Urdu:""",
            
            'option': """Translate the following Bengali option to Urdu. Keep the meaning exactly the same and maintain the cultural context. Only return the Urdu translation, nothing else.

Bengali: {text}
Urdu:"""
        }
    
    def setup_current_key(self):
        """Setup Gemini with current API key"""
        try:
            api_key = KEY_LIST[self.current_key_index]
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(MODEL)
            logging.info(f"Using API key index {self.current_key_index}")
        except Exception as e:
            logging.error(f"Error setting up key {self.current_key_index}: {e}")
            raise
    
    def switch_key(self):
        """Switch to next API key"""
        self.current_key_index = (self.current_key_index + 1) % len(KEY_LIST)
        if self.current_key_index == 0:
            logging.warning("Cycled through all keys, starting from beginning")
        
        self.setup_current_key()
        logging.info(f"Switched to API key index {self.current_key_index}")
        time.sleep(5)  # Brief pause after switching
    
    def translate_text(self, text: str, text_type: str = 'caption', max_retries: int = 3) -> str:
        """Translate text with rate limit management"""
        if not text.strip():
            return text
            
        prompt = self.translation_prompts[text_type].format(text=text)
        
        # Rate limit management: 10 RPM = max 1 request per 6 seconds
        # Adding buffer time to be safe + track usage
        current_time = time.time()
        if self.last_request_time > 0:
            time_since_last = current_time - self.last_request_time
            if time_since_last < 7:  # Ensure 7 seconds between requests
                sleep_time = 7 - time_since_last
                logging.info(f"Rate limit protection: sleeping {sleep_time:.1f}s")
                time.sleep(sleep_time)
        
        self.requests_count += 1
        self.daily_requests += 1
        self.last_request_time = time.time()
        
        for attempt in range(max_retries):
            for key_attempt in range(len(KEY_LIST)):
                try:
                    response = self.model.generate_content(prompt)
                    
                    if response and hasattr(response, 'text'):
                        translated = response.text.strip()
                        if translated:
                            # Estimate token usage (rough: 1 token ≈ 4 chars for Bengali/Urdu)
                            estimated_tokens = (len(text) + len(translated)) // 4
                            self.tokens_used += estimated_tokens
                            
                            # Log rate limit status periodically
                            if self.requests_count % 5 == 0:
                                logging.info(f"Rate Limit Status - Requests: {self.requests_count}, Tokens: ~{self.tokens_used}, Daily: {self.daily_requests}")
                            
                            logging.info(f"Translated ({estimated_tokens} tokens): {text[:50]}...")
                            return translated
                    
                    logging.warning(f"Empty response for text: {text[:50]}...")
                    return text  # Return original if empty response
                    
                except Exception as e:
                    error_msg = str(e).lower()
                    
                    # Check for rate limit or quota exceeded errors
                    if any(term in error_msg for term in ['quota', 'rate limit', '429', 'resource_exhausted']):
                        logging.warning(f"Rate limit hit, switching key. Error: {e}")
                        self.switch_key()
                        time.sleep(30)  # Longer wait after rate limit
                        continue
                    
                    # Other errors
                    logging.error(f"Translation error (attempt {attempt+1}/{max_retries}): {e}")
                    if attempt < max_retries - 1:
                        time.sleep(10)  # Longer backoff
                        break
                    else:
                        logging.error(f"Failed to translate after {max_retries} attempts: {text[:50]}...")
                        return text  # Return original text if all retries failed
        
        return text  # Fallback to original text

class DataTranslator:
    def __init__(self):
        self.translator = GeminiTranslator()
        self.source_dir = "/mnt/wwn-0x500a0751e14d807e-part2/Labib/Labib Folder/Labib/Research/BanglaVerse Experiments/BanglaVerse_V2/data"
        self.target_dir = "/mnt/wwn-0x500a0751e14d807e-part2/Labib/Labib Folder/Labib/Research/BanglaVerse Experiments/BanglaVerse_V2/data_others/data_urdu"
        
        # Create target directory if it doesn't exist
        Path(self.target_dir).mkdir(parents=True, exist_ok=True)
        
        # Categories to process
        self.categories = ['culture', 'food', 'history', 'media_and_movies', 
                          'national_achievements', 'nature', 'personalities', 'politics', 'sports']
        
        # Checkpoint file to track progress
        self.checkpoint_file = Path(self.target_dir) / "translation_checkpoint.json"
        self.checkpoint_data = self.load_checkpoint()
    
    def load_checkpoint(self) -> Dict:
        """Load checkpoint data to resume from where we left off"""
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                    checkpoint = json.load(f)
                logging.info(f"Loaded checkpoint: {checkpoint}")
                return checkpoint
            except Exception as e:
                logging.error(f"Error loading checkpoint: {e}")
        
        # If no checkpoint exists, auto-detect completed files
        completed_files = self.auto_detect_completed_files()
        initial_checkpoint = {
            "completed_files": completed_files, 
            "current_category": None, 
            "current_file": None, 
            "current_index": 0
        }
        
        if completed_files:
            logging.info(f"Auto-detected {len(completed_files)} completed files: {completed_files}")
            # Save the initial checkpoint
            with open(self.checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(initial_checkpoint, f, indent=2)
        
        return initial_checkpoint
    
    def auto_detect_completed_files(self) -> List[str]:
        """Auto-detect which files are already completed by checking if they exist and have content"""
        completed_files = []
        
        file_types = ['captions.json', 'qa_pairs.json', 'commonsense_reasoning.json']
        
        for category in self.categories:
            for file_type in file_types:
                filename = f"{category}_{file_type}"
                target_file = Path(self.target_dir) / category / "annotations" / filename
                
                if target_file.exists():
                    try:
                        # Check if file has valid content
                        with open(target_file, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        
                        if data and len(data) > 0:  # Non-empty file with content
                            file_key = f"{category}_{filename}"
                            completed_files.append(file_key)
                            logging.info(f"Auto-detected completed file: {file_key} ({len(data)} items)")
                    except Exception as e:
                        logging.warning(f"Could not read {target_file}: {e}")
        
        return completed_files
    
    def save_checkpoint(self, category: str = None, filename: str = None, index: int = 0):
        """Save current progress to checkpoint"""
        try:
            # Update the current checkpoint data
            self.checkpoint_data.update({
                "current_category": category,
                "current_file": filename,
                "current_index": index,
                "timestamp": time.time()
            })
            
            with open(self.checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(self.checkpoint_data, f, indent=2)
                
        except Exception as e:
            logging.error(f"Error saving checkpoint: {e}")
    
    def is_file_completed(self, category: str, filename: str) -> bool:
        """Check if a file is already completed"""
        file_key = f"{category}_{filename}"
        return file_key in self.checkpoint_data.get("completed_files", [])
    
    def mark_file_completed(self, category: str, filename: str):
        """Mark a file as completed"""
        file_key = f"{category}_{filename}"
        if file_key not in self.checkpoint_data.get("completed_files", []):
            self.checkpoint_data["completed_files"].append(file_key)
            self.save_checkpoint()
    
    def get_resume_point(self, category: str, filename: str, data_length: int) -> int:
        """Get the index to resume from for partial files"""
        current_cat = self.checkpoint_data.get("current_category")
        current_file = self.checkpoint_data.get("current_file")
        
        if current_cat == category and current_file == filename:
            resume_index = self.checkpoint_data.get("current_index", 0)
            logging.info(f"Resuming {category}_{filename} from index {resume_index}/{data_length}")
            return resume_index
        
        return 0
    
    def save_incremental_progress(self, category: str, filename: str, translated_data: List[Dict], current_index: int):
        """Save progress incrementally after each translation"""
        target_file = Path(self.target_dir) / category / "annotations" / filename
        
        try:
            # Save the current translated data
            with open(target_file, 'w', encoding='utf-8') as f:
                json.dump(translated_data, f, ensure_ascii=False, indent=2)
            
            # Update checkpoint
            self.save_checkpoint(category, filename, current_index)
            
            logging.info(f"Saved progress: {len(translated_data)} items to {filename}")
            
        except Exception as e:
            logging.error(f"Error saving incremental progress: {e}")
    
    def translate_captions(self, data: List[Dict], category: str = None) -> List[Dict]:
        """Translate caption data with resume capability and incremental saving"""
        filename = f"{category}_captions.json"
        target_file = Path(self.target_dir) / category / "annotations" / filename
        
        # Initialize translated_data with the same length as source data
        translated_data = [None] * len(data)
        
        # Load existing translations if file exists
        if target_file.exists():
            try:
                with open(target_file, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                
                # Map existing translations to correct positions based on image_id
                for existing_item in existing_data:
                    if 'image_id' in existing_item:
                        # Find matching item in source data
                        for i, source_item in enumerate(data):
                            if source_item.get('image_id') == existing_item['image_id']:
                                translated_data[i] = existing_item
                                break
                
                logging.info(f"Loaded existing translations from {filename}")
            except Exception as e:
                logging.warning(f"Could not load existing file: {e}")
        
        # Find the first untranslated item
        start_from = 0
        for i, item in enumerate(translated_data):
            if item is None:
                start_from = i
                break
        else:
            # All items are translated
            logging.info(f"All captions already translated in {filename}")
            return [item for item in translated_data if item is not None]
        
        logging.info(f"Resuming captions from item {start_from + 1}/{len(data)}")
        
        # Translate remaining items
        for i in range(start_from, len(data)):
            if translated_data[i] is not None:
                continue  # Skip already translated items
                
            item = data[i]
            translated_item = item.copy()
            
            # Translate caption
            if 'caption' in item:
                translated_item['caption'] = self.translator.translate_text(item['caption'], 'caption')
            
            # Store the translated item
            translated_data[i] = translated_item
            
            # Create clean data list (no None values) for saving
            clean_data = [item for item in translated_data if item is not None]
            
            # Save progress after EACH translation (rate limit friendly)
            self.save_incremental_progress(category, filename, clean_data, i + 1)
            
            logging.info(f"Completed caption {i + 1}/{len(data)} - Rate limit safe progress")
        
        # Return clean data without None values
        return [item for item in translated_data if item is not None]
    
    def translate_qa_pairs(self, data: List[Dict], category: str = None) -> List[Dict]:
        """Translate QA pairs data with resume capability and incremental saving"""
        filename = f"{category}_qa_pairs.json"
        target_file = Path(self.target_dir) / category / "annotations" / filename
        
        # Initialize translated_data with the same length as source data
        translated_data = [None] * len(data)
        
        # Load existing translations if file exists
        if target_file.exists():
            try:
                with open(target_file, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                
                # Map existing translations to correct positions
                for i, item in enumerate(existing_data):
                    if i < len(translated_data) and item:
                        translated_data[i] = item
                
                logging.info(f"Loaded existing translations from {filename}")
            except Exception as e:
                logging.warning(f"Could not load existing file: {e}")
        
        # Find the first untranslated item
        start_from = 0
        for i, item in enumerate(translated_data):
            if item is None:
                start_from = i
                break
        else:
            # All items are translated
            logging.info(f"All QA pairs already translated in {filename}")
            return [item for item in translated_data if item is not None]
        
        logging.info(f"Resuming QA pairs from item {start_from + 1}/{len(data)}")
        
        # Translate remaining items
        for i in range(start_from, len(data)):
            if translated_data[i] is not None:
                continue  # Skip already translated items
            
            item = data[i]
            translated_item = item.copy()
            
            # Translate question (1 API call)
            if 'question' in item:
                translated_item['question'] = self.translator.translate_text(item['question'], 'question')
            
            # Translate options (multiple API calls - rate limit critical)
            if 'options' in item and isinstance(item['options'], list):
                translated_options = []
                for option in item['options']:
                    translated_option = self.translator.translate_text(option, 'option')
                    translated_options.append(translated_option)
                translated_item['options'] = translated_options
            
            # Translate answer (1 API call)
            if 'answer' in item:
                translated_item['answer'] = self.translator.translate_text(item['answer'], 'answer')
            
            # Store the translated item
            translated_data[i] = translated_item
            
            # Create clean data list (no None values) for saving
            clean_data = [item for item in translated_data if item is not None]
            
            # Save progress after EACH QA pair (since each pair = 6+ API calls)
            self.save_incremental_progress(category, filename, clean_data, i + 1)
            
            logging.info(f"Completed QA pair {i + 1}/{len(data)} - Rate limit safe progress")
        
        # Return clean data without None values
        return [item for item in translated_data if item is not None]
    
    def translate_commonsense(self, data: List[Dict], category: str = None) -> List[Dict]:
        """Translate commonsense reasoning data with resume capability and incremental saving"""
        filename = f"{category}_commonsense_reasoning.json"
        target_file = Path(self.target_dir) / category / "annotations" / filename
        
        # Initialize translated_data with the same length as source data
        translated_data = [None] * len(data)
        
        # Load existing translations if file exists
        if target_file.exists():
            try:
                with open(target_file, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                
                # Map existing translations to correct positions based on image_id
                for existing_item in existing_data:
                    if 'image_id' in existing_item:
                        # Find matching item in source data
                        for i, source_item in enumerate(data):
                            if source_item.get('image_id') == existing_item['image_id']:
                                translated_data[i] = existing_item
                                break
                
                logging.info(f"Loaded existing translations from {filename}")
            except Exception as e:
                logging.warning(f"Could not load existing file: {e}")
        
        # Find the first untranslated item
        start_from = 0
        for i, item in enumerate(translated_data):
            if item is None:
                start_from = i
                break
        else:
            # All items are translated
            logging.info(f"All commonsense items already translated in {filename}")
            return [item for item in translated_data if item is not None]
        
        logging.info(f"Resuming commonsense from item {start_from + 1}/{len(data)}")
        
        # Translate remaining items
        for i in range(start_from, len(data)):
            if translated_data[i] is not None:
                continue  # Skip already translated items
                
            item = data[i]
            translated_item = item.copy()
            
            # Translate question (1 API call)
            if 'question' in item:
                translated_item['question'] = self.translator.translate_text(item['question'], 'question')
            
            # Translate answer (1 API call)
            if 'answer' in item:
                translated_item['answer'] = self.translator.translate_text(item['answer'], 'answer')
            
            # Store the translated item
            translated_data[i] = translated_item
            
            # Create clean data list (no None values) for saving
            clean_data = [item for item in translated_data if item is not None]
            
            # Save progress after EACH commonsense item (2 API calls per item)
            self.save_incremental_progress(category, filename, clean_data, i + 1)
            
            logging.info(f"Completed commonsense {i + 1}/{len(data)} - Rate limit safe progress")
        
        # Return clean data without None values
        return [item for item in translated_data if item is not None]
    
    def process_category(self, category: str):
        """Process all annotation files for a category"""
        logging.info(f"Processing category: {category}")
        
        source_category_dir = Path(self.source_dir) / category / "annotations"
        target_category_dir = Path(self.target_dir) / category / "annotations"
        
        # Create target directory structure
        target_category_dir.mkdir(parents=True, exist_ok=True)
        
        # Define file mappings
        file_mappings = {
            f"{category}_captions.json": self.translate_captions,
            f"{category}_qa_pairs.json": self.translate_qa_pairs,
            f"{category}_commonsense_reasoning.json": self.translate_commonsense
        }
        
        for filename, translation_func in file_mappings.items():
            source_file = source_category_dir / filename
            target_file = target_category_dir / filename
            
            # Skip if file is already completed
            if self.is_file_completed(category, filename):
                logging.info(f"Skipping already completed file: {filename}")
                continue
            
            if source_file.exists():
                try:
                    logging.info(f"Processing {source_file}")
                    
                    # Load source data
                    with open(source_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # Translate data (pass category for checkpoint tracking)
                    translated_data = translation_func(data, category)
                    
                    # Save translated data
                    with open(target_file, 'w', encoding='utf-8') as f:
                        json.dump(translated_data, f, ensure_ascii=False, indent=2)
                    
                    # Mark file as completed
                    self.mark_file_completed(category, filename)
                    
                    logging.info(f"Completed {filename} - saved to {target_file}")
                    
                except Exception as e:
                    logging.error(f"Error processing {source_file}: {e}")
                    traceback.print_exc()
            else:
                logging.warning(f"Source file not found: {source_file}")
    
    def create_images_symlinks(self, category: str):
        """Create symbolic links for images directory"""
        source_images_dir = Path(self.source_dir) / category / "images"
        target_images_dir = Path(self.target_dir) / category / "images"
        
        if source_images_dir.exists():
            try:
                # Create parent directory
                target_images_dir.parent.mkdir(parents=True, exist_ok=True)
                
                # Create symbolic link if it doesn't exist
                if not target_images_dir.exists():
                    target_images_dir.symlink_to(source_images_dir.resolve())
                    logging.info(f"Created symlink for images: {target_images_dir} -> {source_images_dir}")
                else:
                    logging.info(f"Images symlink already exists: {target_images_dir}")
                    
            except Exception as e:
                logging.error(f"Error creating symlink for {category} images: {e}")
    
    def run_translation(self):
        """Run the complete translation process"""
        logging.info("Starting Bangla to Urdu translation process")
        
        for category in self.categories:
            try:
                # Process annotations
                self.process_category(category)
                
                # Create images symlinks
                self.create_images_symlinks(category)
                
                logging.info(f"Completed category: {category}")
                
            except Exception as e:
                logging.error(f"Error processing category {category}: {e}")
                traceback.print_exc()
                continue
        
        logging.info("Translation process completed!")

def main():
    """Main function"""
    translator = DataTranslator()
    translator.run_translation()

if __name__ == "__main__":
    main()

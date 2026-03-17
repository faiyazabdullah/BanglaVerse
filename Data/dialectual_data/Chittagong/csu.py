#!/usr/bin/env python3
"""
Chittagong Dialect Translation for Bengali CSU (Common Sense Understanding)
Converts standard Bengali CSU question-answer pairs to Chittagong dialect using Gemini API with 5-shot prompting
"""

import json
import os
import requests
import time
import logging
from threading import Lock
from pathlib import Path
import random
from tqdm import tqdm
import google.generativeai as genai
import socket

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('chittagong_csu_translation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# API Configuration
KEY_LIST = [
    "AIzaSyDfh7ulSMNVEpZ6Cr0ZIfaZedDtMHJNf0E",
    "AIzaSyD5fLaCbydNxSZJYxYI8xiJuKwVS2Y0aaI",
    "AIzaSyCC-ry-WKMWDaGVcRwYN_FgcUY5sGKsT1s",
    "AIzaSyATQl6OeSpm5kxe2a0TyZ33jAsw__Jqq6M",
    "AIzaSyBmTEInkG9NJlPVh2NVmb99nu_RPUHweus",
    "AIzaSyAEsZUHrrKYKKVysIin6Y30Pycuv9baGAM",
    "AIzaSyBEZ0Vo7A6CCHqX0cAkK-YDuuLbt4dTI5o",
    "AIzaSyACCt9uU3uaIZbsyqZhDzQtRZJkMCcRBPc",
    "AIzaSyCKD8jtDxQswCSkvXDnXs4M_Q6x9Eb9Wu4",
    "AIzaSyBR_78vU5ADRHjx9l8P3aYvAdca-tMHWRA",
    "AIzaSyAIcqatH5JCbXRpOy2Lfjvahign9t5SgjA",
    "AIzaSyD6IiDvyhVjT-LEFJhmVKujKRzLCTzv6Uw",
    "AIzaSyAZxTLUF7qSEE05IGMnTYz6wqYEze1grZ8",
    "AIzaSyANNb9t69WGzB7pTNwHCbgnBk5GYwYi_Z4",
    "AIzaSyBQjQq5krnZhCoAiIo0C6oF9LZagljdByw",
    "AIzaSyABGhNnAfyJua0DKOz4P17Gvyiqu7SFXqw",
    "AIzaSyBN2JvRAzx-waYpfLO-AdC-HDBSXJFvXJ8",
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
NETWORK_TIMEOUT_SECONDS = 120

# Question prompt template
QUESTION_PROMPT_TEMPLATE = """You are an expert in Bengali dialects. Convert the following standard Bengali question to authentic Chittagong dialect.

CRITICAL: Provide ONLY the Chittagong dialect translation. No asterisks (*), no explanations, no additional text.

Chittagong dialect characteristics for questions:
- Use "গো/গা" instead of "টি/টা" 
- Use "লগে" instead of "সাথে"
- Use "আর" instead of "এবং" 
- Use "গাঁ/গাঁয়" instead of "গ্রাম"
- Use "ত/ত্ত" suffix instead of "তে"
- Use "আইত্তে/আইয়া" instead of "এসে/থেকে"
- Use "যাইতেছে" instead of "যাচ্ছে"
- Use "এগো/এইগো" instead of "এটি/এটা"
- Use "পুরানা" instead of "ঐতিহ্যবাহী"
- Use "এলাকা/এলাকাত" instead of "অঞ্চল/অঞ্চলে"
- Use "গুষ্ঠি" instead of "সম্প্রদায়"

Here are 5 examples for questions:

Standard: ছবিতে দেখা যাচ্ছে সাঁওতাল সম্প্রদায়ের বসন্তকালীন ফুলের উৎসব—এটির নাম কী?
Chittagong: ছবিত দেখা যাইতেছে সাঁওতাল গুষ্ঠির বসন্তকালীন ফুলর উৎসব—এগোর নাম কী?

Standard: ভুনা খিচুড়ি সাধারণত কোন খাবারের সাথে পরিবেশন করা হয়?
Chittagong: ভুনা খিচুড়ি সাধারণত কোন খাবারর লগে দেয়া হয়?

Standard: এই খাবারটি সাধারণত কোন উৎসবে খাওয়া হয়?
Chittagong: এই খাবারগো সাধারণত কোন উৎসবত খাওয়া হয়?

Standard: গ্রামীণ মেলায় এই বাদ্যযন্ত্রটি কেন জনপ্রিয়?
Chittagong: গাঁয়র মেলাত এই বাদ্যযন্ত্রগো কেন জনপ্রিয়?

Standard: এই ঐতিহ্যবাহী খেলাটি কোন অঞ্চলে বেশি দেখা যায়?
Chittagong: এই পুরানা খেলাগো কোন এলাকাত বেশি দেখা যায়?

Translate to Chittagong dialect (ONLY provide the clean translation):
Standard: {text_to_translate}
Chittagong:"""

# Answer prompt template  
ANSWER_PROMPT_TEMPLATE = """You are an expert in Bengali dialects. Convert the following standard Bengali answer to authentic Chittagong dialect.

CRITICAL: Provide ONLY the Chittagong dialect translation. No asterisks (*), no explanations, no additional text.

Chittagong dialect characteristics for answers:
- Use "লগে" instead of "সাথে/সঙ্গে"
- Use "এলাকা/এলাকাত" instead of "অঞ্চল/অঞ্চলে"
- Use "আর" instead of "ও" for "and"
- Use "বইলে" instead of "বলে"
- Keep proper nouns (names, places) unchanged unless they have dialect variations
- Use conversational Chittagong patterns

Here are 5 examples for answers:

Standard: ভাজা বেগুন, ডিম বা ভর্তার সঙ্গে
Chittagong: ভাজা বেগুন, ডিম বা ভর্তার লগে

Standard: সহজ ও সুরেলা বলে
Chittagong: সহজ আর সুরেলা বইলে

Standard: চট্টগ্রাম অঞ্চলে
Chittagong: চট্টগ্রাম এলাকাত

Standard: বাহা পরব
Chittagong: বাহা পরব

Standard: পহেলা বৈশাখ
Chittagong: পহেলা বৈশাখ

Translate to Chittagong dialect (ONLY provide the clean translation):
Standard: {text_to_translate}
Chittagong:"""

class RotatingGeminiCSU:
    def __init__(self, key_list, model_name):
        assert key_list, "Provide at least one API key"
        self.keys = key_list
        self.model_name = model_name
        self.key_index = 0
        self.key_usage_count = {key: 0 for key in self.keys}
        self.failed_keys = set()
        self._configure_current_key()

    def _configure_current_key(self):
        key = self.keys[self.key_index]
        genai.configure(api_key=key)
        print(f"➡️ Using API key index {self.key_index}")
        
        # Configure client settings for better timeout handling
        try:
            # Try to configure timeout settings if available in the client
            socket.setdefaulttimeout(NETWORK_TIMEOUT_SECONDS)
        except Exception as e:
            print(f"⚠️ Could not configure socket timeout: {e}")

    def _advance_key(self):
        old = self.key_index
        self.key_index = (self.key_index + 1) % len(self.keys)
        print(f"🔁 Switching API key: {old} -> {self.key_index}")
        self._configure_current_key()

    def ask_once(self, text_to_translate, text_type="question"):
        """One API attempt with currently configured key. Exceptions propagate."""
        if text_type == "question":
            prompt = QUESTION_PROMPT_TEMPLATE.format(text_to_translate=text_to_translate)
        else:
            prompt = ANSWER_PROMPT_TEMPLATE.format(text_to_translate=text_to_translate)

        # Add timeout handling for content generation
        try:
            model = genai.GenerativeModel(self.model_name)
            resp = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.3,
                    top_k=40,
                    top_p=0.95,
                    max_output_tokens=1024,
                )
            )
        except Exception as e:
            # Handle timeout and connection errors specifically
            error_msg = str(e).lower()
            if any(keyword in error_msg for keyword in ['timeout', 'timed out', 'connection', 'ssl', 'socket']):
                raise Exception(f"Network/timeout error during content generation: {e}")
            else:
                raise e
        
        if hasattr(resp, "text") and resp.text:
            return resp.text.strip()
        elif hasattr(resp, "candidates") and resp.candidates:
            cand_text = resp.candidates[0].content.parts[0].text
            return cand_text.strip()
        else:
            return "⚠️ Empty response"

    def ask_with_rotation(self, text_to_translate, text_type="question", max_attempts_per_example=10, backoff_base=2.0):
        """
        Try up to max_attempts_per_example (across keys). On quota/429 or other errors,
        rotate to the next key and retry the same example.
        Enhanced sleep times to avoid PMR (50/min) and PDR (1000/day) limits.
        """
        attempt = 0
        last_exc = None
        while attempt < max_attempts_per_example:
            attempt += 1
            try:
                current_key = self.keys[self.key_index]
                self.key_usage_count[current_key] += 1
                return self.ask_once(text_to_translate, text_type)
            except Exception as e:
                last_exc = e
                msg = str(e).lower()
                if "429" in msg or "quota" in msg or "rate limit" in msg or "quota exceeded" in msg:
                    print(f"❗ Quota/rate-limit detected on key index {self.key_index}: {e}")
                    self.failed_keys.add(self.keys[self.key_index])
                    self._advance_key()
                    # Sleep for rate limits - reasonable backoff
                    sleep_time = 10 + random.uniform(5, 15)  # 15-25 seconds backoff
                    print(f"⏳ Rate limit - backing off for {sleep_time:.1f}s before retrying")
                    time.sleep(sleep_time)
                    continue
                elif any(keyword in msg for keyword in ['timeout', 'timed out', 'connection', 'ssl', 'socket', 'network']):
                    print(f"❗ Network/timeout error on key index {self.key_index} (attempt {attempt}/{max_attempts_per_example}): {e}")
                    # For network errors, try different key and longer backoff
                    self._advance_key()
                    sleep_time = backoff_base * (3 ** (attempt - 1))  # More aggressive backoff for network issues
                    print(f"⏳ Network issue - backing off for {sleep_time:.1f}s before retrying")
                    time.sleep(min(sleep_time, 180))  # Allow up to 3 minutes for network recovery
                    continue
                # other transient errors: rotate and backoff
                print(f"❗ API call failed on key index {self.key_index} (attempt {attempt}/{max_attempts_per_example}): {e}")
                self._advance_key()
                sleep_time = backoff_base * (2 ** (attempt - 1))
                print(f"⏳ backing off for {sleep_time:.1f}s before retrying this example")
                time.sleep(min(sleep_time, 120))
                continue

        print(f"❌ Failed to translate {text_type} after {max_attempts_per_example} attempts. Last error: {last_exc}")
        return "❌ Failed to translate"

class ChittagongCSUTranslator:
    def __init__(self):
        self.rotating_gemini = RotatingGeminiCSU(KEY_LIST, MODEL)
        logger.info(f"Initialized Chittagong CSU translator with {len(KEY_LIST)} API keys")
    
    def translate_text(self, text, text_type="question", max_retries=3):
        """Translate text to Chittagong dialect using rotating Gemini API"""
        try:
            translated_text = self.rotating_gemini.ask_with_rotation(
                text, 
                text_type=text_type,
                max_attempts_per_example=max_retries,
                backoff_base=2.0
            )
            
            # Clean up the response to get only the translation
            if 'Chittagong:' in translated_text:
                translated_text = translated_text.split('Chittagong:')[-1].strip()
            
            # Remove any extra formatting, asterisks, or explanations
            if translated_text.startswith('**'):
                translated_text = translated_text.lstrip('*').strip()
            
            if translated_text.endswith('**'):
                translated_text = translated_text.rstrip('*').strip()
            
            # Remove explanation sections if they exist
            if '\n\n**' in translated_text:
                translated_text = translated_text.split('\n\n**')[0].strip()
            
            # Remove any remaining asterisks at the beginning
            while translated_text.startswith('*'):
                translated_text = translated_text[1:].strip()
            
            # Remove any remaining asterisks at the end
            while translated_text.endswith('*'):
                translated_text = translated_text[:-1].strip()
            
            if translated_text and not translated_text.startswith("❌"):
                logger.info(f"Translation successful ({text_type})")
                return translated_text
            else:
                logger.error(f"Translation failed or returned error for {text_type}")
                return text  # Return original text if translation fails
                
        except Exception as e:
            logger.error(f"Error in {text_type} translation: {e}")
            return text  # Return original text if translation fails
    
    def process_category(self, category):
        """Process a single category of CSU data"""
        # Input and output paths
        input_file = f"/mnt/wwn-0x500a0751e14d807e-part2/Labib/Labib Folder/Labib/Research/BanglaVerse Experiments/BanglaVerse_V2/data/{category}/annotations/{category}_commonsense_reasoning.json"
        output_dir = f"/mnt/wwn-0x500a0751e14d807e-part2/Labib/Labib Folder/Labib/Research/BanglaVerse Experiments/BanglaVerse_V2/data_dialects/dialect_translation/Chittagong/Results/CSU"
        output_file = f"{output_dir}/{category}_csu_chittagong.json"
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load input data
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                csu_data = json.load(f)
            logger.info(f"Loaded {len(csu_data)} CSU pairs from {category}")
        except FileNotFoundError:
            logger.error(f"Input file not found: {input_file}")
            return
        except Exception as e:
            logger.error(f"Error loading input file: {e}")
            return
        
        # Load existing progress if any
        translated_data = []
        start_index = 0
        
        if os.path.exists(output_file):
            try:
                with open(output_file, 'r', encoding='utf-8') as f:
                    translated_data = json.load(f)
                start_index = len(translated_data)
                logger.info(f"Resuming from index {start_index}")
            except Exception as e:
                logger.warning(f"Could not load existing progress: {e}")
        
        # Process CSU data
        for i, csu_item in enumerate(tqdm(csu_data[start_index:], desc=f"Processing {category} CSU", initial=start_index, total=len(csu_data))):
            current_index = start_index + i
            
            try:
                image_id = csu_item['image_id']
                original_question = csu_item['question']
                original_answer = csu_item['answer']
                
                logger.info(f"Translating CSU [{current_index+1}/{len(csu_data)}]: {original_question[:50]}...")
                
                # Translate question and answer separately
                chittagong_question = self.translate_text(original_question, "question")
                # Short delay between question and answer (PMR 50/min = 1.2s minimum)
                time.sleep(1.5 + random.uniform(0.3, 0.7))  # 1.8-2.2 second delay between Q&A
                chittagong_answer = self.translate_text(original_answer, "answer")
                
                # Create result entry
                result = {
                    'image_id': image_id,
                    'original_question': original_question,
                    'chittagong_question': chittagong_question,
                    'original_answer': original_answer,
                    'chittagong_answer': chittagong_answer
                }
                
                translated_data.append(result)
                
                # Save progress every 5 translations (CSU takes longer)
                if (current_index + 1) % 5 == 0 or (current_index + 1) == len(csu_data):
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(translated_data, f, ensure_ascii=False, indent=2)
                    logger.info(f"Progress saved: {current_index + 1}/{len(csu_data)} CSU pairs completed")
                
                # Rate limiting for CSU (2 API calls per item, so need ~2.4s total per CSU pair)
                # Already have 1.8-2.2s between Q&A, so add small buffer between CSU pairs
                sleep_time = 5 + random.uniform(1, 3)
                print(f"⏳ CSU rate limiting: sleeping for {sleep_time:.1f}s")
                time.sleep(sleep_time)
                
            except KeyboardInterrupt:
                logger.info("Translation interrupted by user")
                break
            except Exception as e:
                logger.error(f"Error processing CSU {current_index}: {e}")
                # Add failed entry
                result = {
                    'image_id': csu_item.get('image_id', f'unknown_{current_index}'),
                    'original_question': csu_item.get('question', ''),
                    'chittagong_question': csu_item.get('question', ''),  # Fallback to original
                    'original_answer': csu_item.get('answer', ''),
                    'chittagong_answer': csu_item.get('answer', '')  # Fallback to original
                }
                translated_data.append(result)
        
        # Final save
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(translated_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"=== CSU Translation completed for {category} ===")
        logger.info(f"Total processed: {len(translated_data)}")
        logger.info(f"Output saved to: {output_file}")
        
        # Print API usage statistics
        self.print_api_stats()
    
    def process_all_categories(self):
        """Process all categories"""
        categories = ['culture', 'food', 'history', 'media_and_movies', 'national_achievements', 
                     'nature', 'personalities', 'politics', 'sports']
        
        for category in categories:
            logger.info(f"\n{'='*50}")
            logger.info(f"Starting CSU translation for category: {category}")
            logger.info(f"{'='*50}")
            
            self.process_category(category)
            
            # Wait between categories
            logger.info("Waiting 1 minute between categories for CSU processing...")
            time.sleep(60)  # 1 minute between categories
    
    def print_api_stats(self):
        """Print API usage statistics"""
        logger.info("=== API Usage Statistics ===")
        logger.info(f"Total API keys: {len(self.rotating_gemini.keys)}")
        logger.info(f"Failed keys: {len(self.rotating_gemini.failed_keys)}")
        logger.info(f"Working keys: {len(self.rotating_gemini.keys) - len(self.rotating_gemini.failed_keys)}")
        logger.info(f"Current key index: {self.rotating_gemini.key_index}")
        
        # Sort by usage count
        sorted_usage = sorted(self.rotating_gemini.key_usage_count.items(), key=lambda x: x[1], reverse=True)
        logger.info("Top 5 used keys:")
        for i, (key, count) in enumerate(sorted_usage[:5], 1):
            logger.info(f"  {i}. ...{key[-8:]} - {count} requests")
        
        # Show failed keys
        if self.rotating_gemini.failed_keys:
            logger.info("Failed keys:")
            for key in self.rotating_gemini.failed_keys:
                logger.info(f"  ...{key[-8:]}")

def main():
    """Main function"""
    import sys
    
    translator = ChittagongCSUTranslator()
    
    if len(sys.argv) > 1:
        category = sys.argv[1].lower()
        if category in ['culture', 'food', 'history', 'media_and_movies', 'national_achievements', 
                       'nature', 'personalities', 'politics', 'sports']:
            translator.process_category(category)
        else:
            print(f"Invalid category: {category}")
            print("Valid categories: culture, food, history, media_and_movies, national_achievements, nature, personalities, politics, sports")
    else:
        # Process all categories
        translator.process_all_categories()

if __name__ == "__main__":
    main()
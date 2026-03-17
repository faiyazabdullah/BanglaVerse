#!/usr/bin/env python3
"""
Barishal Dialect Translation for Bengali VQA (Visual Question Answering)
Converts standard Bengali VQA question-answer pairs to Barishal dialect using Gemini API with 5-shot prompting
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
        logging.FileHandler('barishal_vqa_translation.log'),
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

# VQA prompt template
VQA_PROMPT_TEMPLATE = """You are an expert in Bengali dialects. Convert the following standard Bengali VQA question and options to authentic Barishal dialect.

CRITICAL: Provide ONLY the Barishal dialect translations. No asterisks (*), no explanations, no additional text, no numbering.
Return format: EXACTLY 5 lines - First line is the translated question, then exactly 4 options on separate lines.
Do NOT add any explanations, notes, or extra text.

Barishal dialect characteristics for VQA:
- Use "গাম/গাউম" instead of "গ্রাম" 
- Use "লগে" instead of "সাথে"
- Use "খান/খানা" instead of "টি/টা" 
- Use "কইর্যা/করিয়া" instead of "করে"
- Use "থেইক্যা" instead of "থেকে"
- Use "দেখা যাইতাছে" instead of "দেখা যাচ্ছে"
- Use "এইটা/এইখান" instead of "এটি/এটা"
- Use "পুরান" instead of "ঐতিহ্যবাহী"
- Use "এলাকা/এলাকায়" instead of "অঞ্চল/অঞ্চলে"
- Use "উতসব" instead of "উৎসব"
- Use "আর" instead of "ও" for "and"
- Use "যেইখান" instead of "যা/যেটা"
- Use "বাজাইতে" instead of "বাজাতে"
- Keep proper nouns (places, names) unchanged

Here are 5 examples:

Standard Question: ছবিতে দেখা উৎসবটি কোন সম্প্রদায়ের পরিচয়ের প্রতীক?
Standard Options: সাঁওতাল, গারো, চাকমা, মারমা
Barishal Question: ছবিতে দেখা উতসবখান কোন সম্প্রদায়ের পরিচয়ের প্রতীক?
Barishal Options: সাঁওতাল, গারো, চাকমা, মারমা

Standard Question: এই খাবারটির নাম কী?
Standard Options: ভুনা খিচুরি, পান্তা ভাত, চাপ, পোলাও
Barishal Question: এই খাবারখানের নাম কী?
Barishal Options: ভুনা খিচুরি, পান্তা ভাত, চাপ, পোলাও

Standard Question: কোন খাবারের প্রধান উপাদান ভাত ও ইলিশ মাছ, যা বিশেষভাবে গ্রামীণ বাংলাদেশের সকালে খাওয়া হয়?
Standard Options: ভুনা খিচুরি, পান্তা ইলিশ, চাপ, পোলাও
Barishal Question: কোন খাবারের প্রধান উপাদান ভাত আর ইলিশ মাছ, যেইখান বিশেষভাবে গামের বাংলাদেশের সকালে খাওয়া হয়?
Barishal Options: ভুনা খিচুরি, পান্তা ইলিশ, চাপ, পোলাও

Standard Question: ছবির বাদ্যযন্ত্রটি সাধারণত কোন ধরনের সুর বাজাতে ব্যবহৃত হয়?
Standard Options: লোকগীতি, ক্লাসিক্যাল, আধুনিক, রক
Barishal Question: ছবির বাদ্যযন্ত্রখান সাধারণত কোন ধরনের সুর বাজাইতে ব্যবহার হয়?
Barishal Options: লোকগীতি, ক্লাসিক্যাল, আধুনিক, রক

Standard Question: এই ঐতিহ্যবাহী খেলাটি কোন অঞ্চলে বেশি জনপ্রিয়?
Standard Options: ঢাকা, চট্টগ্রাম, সিলেট, রাজশাহী
Barishal Question: এই পুরান খেলাখান কোন এলাকায় বেশি জনপ্রিয়?
Barishal Options: ঢাকা, চট্টগ্রাম, সিলেট, রাজশাহী

Translate to Barishal dialect (provide question first, then each option on separate lines):
Standard Question: {question_to_translate}
Standard Options: {options_str}
Barishal Question:"""

class RotatingGeminiVQA:
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

    def ask_once(self, question_to_translate, options_to_translate):
        """One API attempt with currently configured key. Exceptions propagate."""
        options_str = ", ".join(options_to_translate)
        prompt = VQA_PROMPT_TEMPLATE.format(
            question_to_translate=question_to_translate, 
            options_str=options_str
        )

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

    def ask_with_rotation(self, question_to_translate, options_to_translate, max_attempts_per_example=10, backoff_base=2.0):
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
                return self.ask_once(question_to_translate, options_to_translate)
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

        print(f"❌ Failed to translate VQA after {max_attempts_per_example} attempts. Last error: {last_exc}")
        return "❌ Failed to translate"

class BarishalVQATranslator:
    def __init__(self):
        self.rotating_gemini = RotatingGeminiVQA(KEY_LIST, MODEL)
        logger.info(f"Initialized Barishal VQA translator with {len(KEY_LIST)} API keys")

    
    def translate_vqa_item(self, question, options, max_retries=3):
        """Translate VQA question and options to Barishal dialect using rotating Gemini API"""
        try:
            translated_text = self.rotating_gemini.ask_with_rotation(
                question, 
                options,
                max_attempts_per_example=max_retries,
                backoff_base=2.0
            )
            
            if translated_text.startswith("❌"):
                logger.error("Translation failed or returned error")
                return question, options  # Return original if translation fails
            
            # Aggressive cleaning of the response
            if 'Barishal Question:' in translated_text:
                translated_text = translated_text.split('Barishal Question:')[-1].strip()
            
            if 'Barishal Options:' in translated_text:
                translated_text = translated_text.split('Barishal Options:')[-1].strip()
            
            # Remove any explanation sections (common patterns)
            cleanup_patterns = [
                '\n\n**',
                '\n\nNote:',
                '\n\nExplanation:',
                '\n\n*',
                '\n\n(Note:',
                '\n\nটীকা:',
                '\n\nব্যাখ্যা:'
            ]
            
            for pattern in cleanup_patterns:
                if pattern in translated_text:
                    translated_text = translated_text.split(pattern)[0].strip()
            
            # Remove asterisks and formatting markers
            translated_text = translated_text.replace('**', '').replace('*', '')
            
            # Clean each line individually 
            lines = translated_text.split('\n')
            cleaned_lines = []
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Remove common prefixes/suffixes
                line = line.lstrip('*').rstrip('*').strip()
                
                # Skip explanation lines
                if any(skip_word in line.lower() for skip_word in ['note:', 'explanation:', 'টীকা:', 'ব্যাখ্যা:']):
                    continue
                    
                cleaned_lines.append(line)
            
            # Rejoin the cleaned lines
            translated_text = '\n'.join(cleaned_lines)
            
            # Parse the response: first line is question, rest are options
            lines = [line.strip() for line in translated_text.split('\n') if line.strip()]
            
            # Debug: log the actual response for troubleshooting
            if len(lines) != 5:
                logger.info(f"Response lines ({len(lines)}): {lines}")
            
            if len(lines) >= 5:  # Question + 4 options
                barishal_question = lines[0]
                barishal_options = lines[1:5]  # Take exactly 4 options
                
                logger.info(f"VQA translation successful")
                return barishal_question, barishal_options
            elif len(lines) == 2:
                # Common case: Question on first line, options on second line (comma-separated)
                barishal_question = lines[0]
                
                # Try to split the second line by commas to get options
                options_line = lines[1]
                if ',' in options_line:
                    options_found = [opt.strip() for opt in options_line.split(',')]
                    if len(options_found) == 4:
                        logger.info(f"VQA translation successful (2-line format)")
                        return barishal_question, options_found
                
                # If that doesn't work, fall through to general parsing
            
            if len(lines) >= 1:
                # Try to extract from a different format
                # Look for the question and options pattern
                barishal_question = lines[0]
                
                # Find options - they might start with numbers or be after certain keywords
                options_found = []
                for line in lines[1:]:
                    # Skip empty lines or explanation markers
                    if not line or line.startswith('**') or 'explanation' in line.lower() or 'note' in line.lower():
                        continue
                    
                    # Handle comma-separated options in a single line
                    if ',' in line and len(options_found) == 0:
                        comma_options = [opt.strip() for opt in line.split(',')]
                        if len(comma_options) == 4:
                            options_found = comma_options
                            break
                    
                    # Add non-empty lines as options until we have 4
                    if len(options_found) < 4:
                        # Clean option text (remove numbering, bullets, etc.)
                        clean_option = line
                        if clean_option.startswith(('1.', '2.', '3.', '4.', 'ক.', 'খ.', 'গ.', 'ঘ.', '•', '-')):
                            clean_option = clean_option[2:].strip()
                        options_found.append(clean_option)
                
                if len(options_found) == 4:
                    logger.info(f"VQA translation successful (alt format)")
                    return barishal_question, options_found
                else:
                    logger.warning(f"Unexpected response format: {len(lines)} lines, {len(options_found)} options found")
                    # Last resort: if we have a question but not enough options, 
                    # use original options with translated question
                    if len(lines) >= 1 and lines[0].strip():
                        logger.info(f"Using translated question with original options")
                        return lines[0].strip(), options
                    # Complete fallback: return original if parsing fails
                    return question, options
            else:
                logger.warning(f"Unexpected response format: {len(lines)} lines")
                # Fallback: return original if parsing fails
                return question, options
                
        except Exception as e:
            logger.error(f"Error in VQA translation: {e}")
            return question, options  # Return original if translation fails
    
    def process_category(self, category):
        """Process a single category of VQA data"""
        # Input and output paths
        input_file = f"/mnt/wwn-0x500a0751e14d807e-part2/Labib/Labib Folder/Labib/Research/BanglaVerse Experiments/BanglaVerse_V2/data/{category}/annotations/{category}_qa_pairs.json"
        output_dir = f"/mnt/wwn-0x500a0751e14d807e-part2/Labib/Labib Folder/Labib/Research/BanglaVerse Experiments/BanglaVerse_V2/data_dialects/dialect_translation/Barishal/Results/VQA"
        output_file = f"{output_dir}/{category}_vqa_barishal.json"
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load input data
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                vqa_data = json.load(f)
            logger.info(f"Loaded {len(vqa_data)} VQA pairs from {category}")
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
        
        # Process VQA pairs
        for i, vqa_item in enumerate(tqdm(vqa_data[start_index:], desc=f"Processing {category} VQA", initial=start_index, total=len(vqa_data))):
            current_index = start_index + i
            
            try:
                image_id = vqa_item['image_id']
                original_question = vqa_item['question']
                original_options = vqa_item['options']
                original_answer = vqa_item['answer']
                
                logger.info(f"Translating VQA [{current_index+1}/{len(vqa_data)}]: {original_question[:50]}...")
                
                # Translate question and options together
                barishal_question, barishal_options = self.translate_vqa_item(original_question, original_options)
                
                # The answer should remain the same as it's one of the options
                # Find the answer in the translated options
                barishal_answer = original_answer
                if original_answer in original_options:
                    answer_index = original_options.index(original_answer)
                    if answer_index < len(barishal_options):
                        barishal_answer = barishal_options[answer_index]
                
                # Create result entry
                result = {
                    'image_id': image_id,
                    'original_question': original_question,
                    'barishal_question': barishal_question,
                    'original_options': original_options,
                    'barishal_options': barishal_options,
                    'original_answer': original_answer,
                    'barishal_answer': barishal_answer
                }
                
                translated_data.append(result)
                
                # Save progress every 5 translations
                if (current_index + 1) % 5 == 0 or (current_index + 1) == len(vqa_data):
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(translated_data, f, ensure_ascii=False, indent=2)
                    logger.info(f"Progress saved: {current_index + 1}/{len(vqa_data)} VQA pairs completed")
                
                # Rate limiting for VQA (1 API call per item, PMR 50/min = 1.2s minimum)
                # Adding small buffer: 1.5-2 seconds between VQA pairs
                sleep_time = 5 + random.uniform(1, 3)
                print(f"⏳ VQA rate limiting: sleeping for {sleep_time:.1f}s")
                time.sleep(sleep_time)
                
            except KeyboardInterrupt:
                logger.info("Translation interrupted by user")
                break
            except Exception as e:
                logger.error(f"Error processing VQA {current_index}: {e}")
                # Add failed entry
                result = {
                    'image_id': vqa_item.get('image_id', f'unknown_{current_index}'),
                    'original_question': vqa_item.get('question', ''),
                    'barishal_question': vqa_item.get('question', ''),  # Fallback to original
                    'original_options': vqa_item.get('options', []),
                    'barishal_options': vqa_item.get('options', []),  # Fallback to original
                    'original_answer': vqa_item.get('answer', ''),
                    'barishal_answer': vqa_item.get('answer', '')  # Fallback to original
                }
                translated_data.append(result)
        
        # Final save
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(translated_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"=== VQA Translation completed for {category} ===")
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
            logger.info(f"Starting VQA translation for category: {category}")
            logger.info(f"{'='*50}")
            
            self.process_category(category)
            
            # Wait between categories
            logger.info("Waiting 30 seconds between categories for VQA processing...")
            time.sleep(30)  # 30 seconds between categories
    
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
    
    translator = BarishalVQATranslator()
    
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

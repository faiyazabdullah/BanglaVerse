# BanglaVerse

## A Benchmark Dataset for Visual Understanding in Bangla with Cultural Awareness

**BanglaVerse** is a curated visual dataset representing the diverse cultural, historical, and socio-political landscape of **Bangladesh**. This dataset is designed for evaluating the visual and reasoning capabilities of **multilingual vision language models**, especially those aligned for **Bangla** and **low-resource language** understanding.

---

## 📂 Dataset Structure

```
BanglaVision/
├── culture/
│   ├── images/
│   │   ├── culture_001.png
│   │   ├── culture_002.png
│   │   └── ...
│   └── annotations/
│       ├── culture_captions.json
│       ├── culture_qa_pairs.json
│       └── culture_commonsense_reasoning.json
├── history/
├── politics/
├── national_achievements/
├── sports/
├── media_and_movies/
├── personalities/
└── food/
```

---

## 📦 Dataset Components

### 🖼️ 1. Captions

Short descriptions of each image in **Bangla**.

```json
{
  "image_id": "food_002",
  "caption": "একটি প্লেটে পরিবেশন করা গরম গরম ইলিশ মাছের সাথে পান্তা ভাত।"
}
```

---

### ❓ 2. Visual Question Answering (VQA)

Task: Answer direct visual questions about an image based on multiple choices.

```json
{
  "image_id": "sports_005",
  "question_bn": "ছবিতে কোন খেলাটি চলছে?",
  "options_bn": ["ক্রিকেট", "ফুটবল", "হ্যান্ডবল", "কাবাডি"],
  "answer_bn": "কাবাডি"
}
```

---

### 🧠 3. Commonsense Reasoning Tasks

Task: Answer contextual and culturally relevant questions about the image.

```json
{
  "image_id": "media_002",
  "question": "ছবির এই টেলিভিশন নাটকটি কোন জনপ্রিয় লেখকের রচনায় নির্মিত?",
  "answer": "হুমায়ূন আহমেদ"
}
```

---

## 🙏 Acknowledgements

This project is inspired by the rich heritage, resilience, and identity of Bangladesh. It aims to bridge the gap between global AI systems and the narratives of underrepresented cultures.

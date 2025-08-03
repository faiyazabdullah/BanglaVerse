# 🇧🇩 BanglaVerse: A Grounded Benchmark for Visual Reasoning in Bangla with Cultural and Historical Awareness

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

```

---

### ❓ 2. Visual Question Answering (VQA)

Task: Answer direct visual questions about an image.

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

```

## 🙏 Acknowledgements

This project is inspired by the rich heritage, resilience, and identity of Bangladesh. It aims to bridge the gap between global AI systems and the narratives of underrepresented cultures.

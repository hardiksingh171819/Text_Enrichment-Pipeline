# Text Enrichment Pipeline (Lightweight NLP)

## Overview
This project is a lightweight Natural Language Processing (NLP) pipeline designed to extract meaningful insights from unstructured text. It combines multiple NLP techniques to generate a structured and interpretable output in both JSON and HTML formats.

The pipeline is built for quick analysis of textual data, enabling users to summarize content, identify key entities, and understand sentiment in a single workflow.

---

## Key Features

- Text Summarization using TextRank (extractive summarization)
- Named Entity Recognition (NER) using spaCy
- Sentiment Analysis using VADER
- Automated JSON output for structured data
- Interactive HTML report for easy visualization

---

## Use Cases

- Customer feedback analysis  
- Social media sentiment monitoring  
- Document summarization  
- News/article analysis  
- Preprocessing pipeline for downstream ML/NLP tasks  

---

## Project Workflow

```
Input Text
│
▼
Summarization (TextRank)
│
▼
Named Entity Recognition (spaCy)
│
▼
Sentiment Analysis (VADER)
│
▼
Structured Output (JSON + HTML Report)
```
---

## Technologies Used

- Python  
- spaCy (NER)  
- sumy (TextRank summarization)  
- VADER Sentiment Analysis  
- argparse (CLI interface)  

---

## Installation


### Install dependencies

pip install sumy spacy vaderSentiment
python -m spacy download en_core_web_sm


---

## How to Run

### Option 1: With input file

python main.py --input sample.txt


### Option 2: Manual input

python main.py


(Paste your text and press Enter twice)

---

## Optional Arguments

| Argument        | Description |
|----------------|------------|
| `--input` / `-i` | Input text file |
| `--output` / `-o` | Custom output filename |
| `--sentences` / `-s` | Number of summary sentences (default: 3) |
| `--no-html` | Skip HTML report generation |

---

## Output

### JSON Output
Contains:
- Summary  
- Named Entities  
- Sentiment scores  
- Original text  

### HTML Report
Includes:
- Clean summary section  
- Entity table  
- Sentiment breakdown  
- Original text view  

---

## Example Output Structure

```json
{
  "summary": "...",
  "entities": [
    {"text": "Apple", "label": "ORG"}
  ],
  "sentiment": {
    "label": "positive",
    "scores": {
      "neg": 0.0,
      "neu": 0.8,
      "pos": 0.2,
      "compound": 0.5
    }
  }
}
```

## Key Design Decisions
Lightweight architecture for quick execution
CLI-based interface for flexibility
Modular functions for easy extension
Automatic model handling (spaCy download if missing)
Dual output format for both technical and business users

---

## Potential Improvements
Add abstractive summarization (e.g., transformers)
Multi-language support
Topic modeling integration
REST API deployment
Real-time streaming input support

---

## Author
Hardik Singh
MSc Data Management & Artificial Intelligence

---

## Summary
This project demonstrates how multiple NLP techniques can be combined into a single pipeline to transform raw text into actionable insights, making it valuable for both technical and business applications.

---

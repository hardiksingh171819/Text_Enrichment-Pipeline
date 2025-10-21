"""
Text Enrichment Pipeline (Light Version)
-------------------------------------------------
Performs:
 - Summarization (TextRank via sumy)
 - Named Entity Recognition (spaCy)
 - Sentiment Analysis (VADER)
Outputs both JSON and HTML report.
"""

import argparse
import json
from pathlib import Path
from datetime import datetime

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer
import spacy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer



def summarize_text(text, sentences_count=3):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = TextRankSummarizer()
    summary_sentences = summarizer(parser.document, sentences_count)
    summary = " ".join(str(s) for s in summary_sentences)
    return summary if summary.strip() else "(No clear summary generated.)"

def extract_entities(text, nlp):
    doc = nlp(text)
    return [{"text": ent.text, "label": ent.label_} for ent in doc.ents]

def classify_sentiment(text, analyzer):
    scores = analyzer.polarity_scores(text)
    compound = scores["compound"]
    label = "positive" if compound >= 0.05 else "negative" if compound <= -0.05 else "neutral"
    return {"label": label, "scores": scores}

def save_json(obj, out_path):
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def make_html_report(results, out_html_path):
    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Text Enrichment Report</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.5; }}
pre {{ background:#f7f7f7; padding:12px; border-radius:6px; }}
.box {{ padding:10px; border:1px solid #ddd; border-radius:6px; margin-bottom:16px; }}
.entities td {{ padding:6px; border-bottom:1px solid #eee; }}
</style></head><body>
<h1>Text Enrichment Report</h1>
<div class="box"><h2>Summary</h2><p>{results['summary']}</p></div>
<div class="box"><h2>Named Entities</h2>
<table class="entities"><tr><th>Entity</th><th>Label</th></tr>
{''.join(f'<tr><td>{e["text"]}</td><td>{e["label"]}</td></tr>' for e in results['entities'])}
</table></div>
<div class="box"><h2>Sentiment</h2>
<p><strong>Label:</strong> {results['sentiment']['label']}</p>
<pre>{json.dumps(results['sentiment']['scores'], indent=2)}</pre></div>
<div class="box"><h2>Original Text</h2><pre>{results['original_text']}</pre></div>
</body></html>"""
    with open(out_html_path, "w", encoding="utf-8") as f:
        f.write(html)

# ---------------------------
# Main function
# ---------------------------

def main():
    parser = argparse.ArgumentParser(description="Lightweight text enrichment pipeline.")
    parser.add_argument("--input", "-i", type=str, help="Input text file (optional)")
    parser.add_argument("--output", "-o", type=str, default="", help="Output file prefix (optional)")
    parser.add_argument("--sentences", "-s", type=int, default=3, help="Number of sentences for summary")
    parser.add_argument("--no-html", action="store_true", help="Skip HTML report")
    args = parser.parse_args()

    # Read text
    if args.input:
        text = Path(args.input).read_text(encoding="utf-8")
    else:
        print("Paste your text (end with an empty line):")
        lines = []
        while True:
            line = input()
            if not line.strip(): break
            lines.append(line)
        text = "\n".join(lines)

    # Timestamped output names
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_name = args.output or f"results_{timestamp}.json"
    html_name = f"report_{timestamp}.html"

    print(" Loading spaCy model...")
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("Downloading spaCy model...")
        from spacy.cli import download
        download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")

    analyzer = SentimentIntensityAnalyzer()

    print(" Summarizing...")
    summary = summarize_text(text, args.sentences)

    print(" Extracting entities...")
    entities = extract_entities(text, nlp)

    print(" Analyzing sentiment...")
    sentiment = classify_sentiment(text, analyzer)

    results = {
        "summary": summary,
        "entities": entities,
        "sentiment": sentiment,
        "original_text": text.strip()
    }

    save_json(results, json_name)
    print(f"✅ JSON saved as {json_name}")

    if not args.no_html:
        make_html_report(results, html_name)
        print(f"✅ HTML report saved as {html_name}")

if __name__ == "__main__":
    main()


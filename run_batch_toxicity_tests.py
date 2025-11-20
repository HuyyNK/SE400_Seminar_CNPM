"""Batch testing utility for the Hybrid Toxic Content Classifier.

Usage examples:
    python run_batch_toxicity_tests.py
    python run_batch_toxicity_tests.py --input-file my_sentences.txt
    python run_batch_toxicity_tests.py --text "Custom sentence to test"
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import List, Sequence

import joblib
import pandas as pd
import pickle

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize
except ImportError as exc:
    raise SystemExit("Please install nltk to run this script: pip install nltk") from exc

from CrawlData.model import ToxicPhraseDetector


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_ROOT / "saved_models"
SLANG_PATH = PROJECT_ROOT / "Data" / "slang.csv"


lemmatizer: WordNetLemmatizer | None = None
stop_words: set[str] | None = None


def ensure_nltk_resources():
    """Download required NLTK corpora if absent and init globals."""
    for resource in ["stopwords", "punkt", "punkt_tab", "wordnet", "averaged_perceptron_tagger"]:
        try:
            nltk.data.find(f"corpora/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)

    global lemmatizer, stop_words
    if lemmatizer is None:
        lemmatizer = WordNetLemmatizer()
    if stop_words is None:
        stop_words = set(stopwords.words("english"))


def normalize_repeated_chars(text: str, max_repeats: int = 2) -> str:
    pattern = re.compile(rf"([a-zA-Z])\1{{{max_repeats},}}")
    return pattern.sub(lambda m: m.group(1) * max_repeats, text)


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = normalize_repeated_chars(text)
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"@\w+|#\w+", "", text)
    text = re.sub(r"&\w+;|&#\d+;", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_text(text: str) -> str:
    try:
        if lemmatizer is None or stop_words is None:
            return text
        tokens = word_tokenize(text)
        tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and len(word) > 2]
        return " ".join(tokens)
    except Exception:
        return text


class HybridToxicClassifier:
    """Hybrid classifier combining Rule-based filter + ML model with tiered labels."""

    DEFAULT_SPAM_KEYWORDS = (
        # Common spam triggers
        "buy now",
        "order now",
        "click here",
        "limited offer",
        "flash sale",
        "discount",
        "promo code",
        "coupon",
        "free shipping",
        "subscribe now",
        "follow my channel",
        "visit our website",
        "visit my channel",
        "hotline",
        "liên hệ",
        "mua ngay",
        "giảm giá",
        "khuyến mãi",
        "ưu đãi",
        "zalo",
        "telegram",
        "inbox",
        "dm for price",
        "săn sale",
        "deal độc quyền",
        # Scam/prize/urgency keywords
        "congratulations",
        "you won",
        "you have won",
        "free iphone",
        "free phone",
        "claim now",
        "claim here",
        "click to claim",
        "winner",
        "selected winner",
        "lucky winner",
        "make $",
        "earn $",
        "make money",
        "from home",
        "zero effort",
        "no effort",
        "miracle pill",
        "lose weight",
        "lose 20lbs",
        "guaranteed",
        "100% guaranteed",
        "urgent",
        "account compromised",
        "verify now",
        "verify identity",
        "verify account",
        "suspended account",
        "unusual activity",
        "confirm identity",
        "security alert",
        "act now",
        "limited time",
        "expires soon",
        "last chance",
        "dont miss",
        "risk free",
        "no risk",
        "money back",
        "refund guarantee",
    )
    URL_PATTERN = re.compile(r"(https?://|www\.|\.com\b|\.vn\b|\.net\b|\[link\])", re.IGNORECASE)
    PHONE_PATTERN = re.compile(r"(?:\+?\d[\s-]?){7,}")
    REPEATED_EXCLAMATION_PATTERN = re.compile(r"!{2,}")
    MONEY_PATTERN = re.compile(r"\$\d+|\d+\s*(?:đô|dollar|usd|vnd|đồng)", re.IGNORECASE)
    CAPS_WORDS_PATTERN = re.compile(r"\b[A-Z]{4,}\b")

    def __init__(
        self,
        ml_model,
        vectorizer,
        rule_detector=None,
        *,
        warning_threshold: float = 0.6,
        violation_threshold: float | None = 0.8,
        spam_keywords=None,
    ):
        if violation_threshold is None:
            violation_threshold = 0.8
        if warning_threshold >= violation_threshold:
            raise ValueError("warning_threshold must be lower than violation_threshold")

        self.ml_model = ml_model
        self.vectorizer = vectorizer
        self.rule_detector = rule_detector
        self.warning_threshold = warning_threshold
        self.violation_threshold = violation_threshold
        self.ml_threshold = violation_threshold
        self.spam_keywords = tuple(spam_keywords) if spam_keywords else self.DEFAULT_SPAM_KEYWORDS

    def _detect_spam(self, raw_text: str):
        """Enhanced heuristic-based spam/scam detector forcing VIOLATION."""
        text_lower = raw_text.lower()
        
        # Check for spam keywords
        for keyword in self.spam_keywords:
            if keyword in text_lower:
                return f"keyword:{keyword}"
        
        # Check for URLs/links
        if self.URL_PATTERN.search(raw_text):
            return "contains_link"
        
        # Check for phone numbers
        if self.PHONE_PATTERN.search(raw_text):
            return "contact_number"
        
        # Check for excessive punctuation (2+ exclamation marks)
        if self.REPEATED_EXCLAMATION_PATTERN.search(raw_text):
            return "excessive_punctuation"
        
        # Check for money mentions (common in scams)
        if self.MONEY_PATTERN.search(raw_text):
            return "money_mention"
        
        # Check for excessive capital letters (common in spam)
        caps_words = self.CAPS_WORDS_PATTERN.findall(raw_text)
        if len(caps_words) >= 2:  # 2 or more ALL CAPS words
            return "excessive_caps"
        
        return None

    def _label_from_probability(self, probability: float):
        if probability > self.violation_threshold:
            return "VIOLATION", True
        if probability >= self.warning_threshold:
            return "WARNING", False
        return "SAFE", False

    def predict(self, text, return_details=False):
        rule_phrases = []
        spam_indicator = self._detect_spam(text)
        if spam_indicator:
            return {
                "text": text,
                "is_violation": True,
                "label": "VIOLATION",
                "method": "spam_filter",
                "ml_probability": None,
                "confidence": 0.92,
                "toxic_phrases": [],
                "spam_indicator": spam_indicator,
                "details": "Detected promotional / spam content",
            }

        if self.rule_detector is not None:
            try:
                rule_result = self.rule_detector.detect(text, return_details=True)
                if rule_result.get("is_toxic", False):
                    return {
                        "text": text,
                        "is_violation": True,
                        "label": "VIOLATION",
                        "method": "rule_based",
                        "ml_probability": None,
                        "confidence": 0.95,
                        "toxic_phrases": rule_result.get("toxic_phrases", []),
                        "spam_indicator": None,
                        "details": "Detected by rule-based filter",
                    }
                rule_phrases = rule_result.get("toxic_phrases", [])
            except Exception as exc:  # pragma: no cover - defensive logging
                print(f"Rule detector error: {exc}")

        cleaned = clean_text(text)
        processed = preprocess_text(cleaned)
        vectorized = self.vectorizer.transform([processed])
        ml_probability = float(self.ml_model.predict_proba(vectorized)[0][1])

        label, is_violation = self._label_from_probability(ml_probability)
        confidence = ml_probability if label != "SAFE" else (1 - ml_probability)
        details = (
            f"Prob={ml_probability:.4f} | tiers -> SAFE < {self.warning_threshold:.2f}, "
            f"WARNING [{self.warning_threshold:.2f}, {self.violation_threshold:.2f}], "
            f"VIOLATION > {self.violation_threshold:.2f}"
        )
        return {
            "text": text,
            "is_violation": is_violation,
            "label": label,
            "method": "ml_model",
            "ml_probability": ml_probability,
            "confidence": float(confidence),
            "toxic_phrases": rule_phrases,
            "spam_indicator": None,
            "details": details,
            "risk_level": label,
        }


def load_hybrid_classifier():
    model_path = MODEL_DIR / "naive_bayes_tuned_balanced.pkl"
    vectorizer_path = MODEL_DIR / "tfidf_vectorizer.pkl"
    metadata_path = MODEL_DIR / "hybrid_model_metadata_optimized.pkl"

    ml_model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)

    warning_threshold = metadata.get("policy_warning_threshold", 0.6)
    violation_threshold = metadata.get("policy_violation_threshold", 0.8)

    rule_detector = None
    if SLANG_PATH.exists():
        rule_detector = ToxicPhraseDetector(SLANG_PATH)

    classifier = HybridToxicClassifier(
        ml_model=ml_model,
        vectorizer=vectorizer,
        rule_detector=rule_detector,
        warning_threshold=warning_threshold,
        violation_threshold=violation_threshold,
    )

    print("✓ Hybrid classifier rebuilt from saved artifacts")
    print(f"  Model: {metadata['model_name']}")
    print(f"  F1-score: {metadata['f1_score']:.4f}")
    print(f"  Warning threshold: {warning_threshold}")
    print(f"  Violation threshold: {violation_threshold}")
    print(f"  Rule-based filter: {rule_detector is not None}")
    print()
    return classifier, metadata


def iter_inputs(args: argparse.Namespace) -> List[str]:
    if args.text:
        return [args.text]
    if args.input_file:
        path = Path(args.input_file)
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")
        with path.open("r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        if not lines:
            raise ValueError("Input file is empty after stripping whitespace")
        return lines
    return [
        "You are a complete moron, I can't believe how stupid you are.",
        "Nobody loves you, you should just disappear and do everyone a favor.",
        "Shut your mouth, you useless piece of trash.",
        "Your opinion is worthless, just like your life.",
        "You're pathetic and everyone is laughing at you behind your back.",
        "Are you mentally challenged or just pretending to be this dumb?",
        "Go back to your own country, you don't belong here.",
        "Women are too emotional to handle leadership roles, stick to the kitchen.",
        "People like you are a disease to our society.",
        "Look at you, you're so ugly and gross, typical of your race.",
        "I hate [Group Name], they ruin everything they touch.",
        "Being gay is unnatural and you should be ashamed of yourself.",
        "I know where you live, and you better watch your back.",
        "If you say that again, I'm going to break your legs.",
        "I hope you get hit by a bus tomorrow.",
        "Keep talking and see what happens to you and your family.",
        "I'm going to find you and make you pay for this.",
        "You deserve to be beaten up for being so annoying.",
        "Hey baby, why don't you send me some naughty photos?",
        "You look like you'd be really good in bed, how much for a night?",
        "Stop playing hard to get, I know you want it.",
        "Nice body, I'd love to see what's under those clothes.",
        "Why are you such a prude? Just show me a little skin.",
        "CONGRATULATIONS! You won a free iPhone 15. Click here to claim: http://scam.me",
        "Make $5000 from home with zero effort! Ask me how!",
        "Lose 20lbs in 2 days with this miracle pill! 100% guaranteed!",
        "URGENT: Your bank account has been compromised. Verify identity now.",
        "Buy now! Limited offer!!! Visit our website for insane discounts!!!",
        "You're actually pretty smart... for someone who didn't go to college.",
        "I love how you just wear anything, even if it doesn't flatter your figure.",
        "Wow, you speak English so well for a foreigner.",
        "I love this beautiful day!",
        "This movie was amazing, highly recommend it!",
        "Thank you for your help, I really appreciate it!",
        "What a wonderful experience, I had a great time!",
    ]


def run_batch(classifier, sentences: Sequence[str]) -> pd.DataFrame:
    rows = []
    for idx, text in enumerate(sentences, start=1):
        result = classifier.predict(text)
        ml_prob = result.get("ml_probability")
        rows.append(
            {
                "#": idx,
                "Text": text[:60] + ("..." if len(text) > 60 else ""),
                "Label": result["label"],
                "Method": result["method"],
                "Probability": f"{ml_prob:.4f}" if ml_prob is not None else "N/A",
                "Confidence": f"{result['confidence']:.4f}",
                "SpamIndicator": result.get("spam_indicator") or "-",
            }
        )

        print(f"Test #{idx}")
        badge = "🔴" if result["label"] == "VIOLATION" else ("🟠" if result["label"] == "WARNING" else "🟢")
        print(f"Text: {text}")
        print(f"Label: {result['label']} {badge}")
        print(f"Method: {result['method']}")
        if ml_prob is not None:
            print(f"ML Probability: {ml_prob:.4f} ({ml_prob * 100:.2f}%)")
        print(f"Confidence: {result['confidence']:.4f} ({result['confidence'] * 100:.2f}%)")
        if result.get("spam_indicator"):
            print(f"Spam Indicator: {result['spam_indicator']}")
        if result.get("toxic_phrases"):
            print(f"⚠️ Toxic Phrases: {', '.join(result['toxic_phrases'])}")
        print("-" * 80)

    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame):
    print("=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print(df.to_string(index=False))
    print()

    total = len(df)
    tier_counts = df["Label"].value_counts().to_dict()
    print("=" * 80)
    print("STATISTICS")
    print("=" * 80)
    print(f"Total tests: {total}")
    for tier in ["VIOLATION", "WARNING", "SAFE"]:
        if tier in tier_counts:
            count = tier_counts[tier]
            print(f"{tier}: {count} ({count / total * 100:.1f}%)")
    print("=" * 80)


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch toxicity tester")
    parser.add_argument("--input-file", help="Path to a text file (one sentence per line)")
    parser.add_argument("--text", help="Single sentence to classify")
    parser.add_argument("--save-json", help="Optional path to export JSON results")
    return parser.parse_args(argv)


def main(argv: Sequence[str]):
    ensure_nltk_resources()
    args = parse_args(argv)
    classifier, metadata = load_hybrid_classifier()
    sentences = iter_inputs(args)
    df = run_batch(classifier, sentences)
    summarize(df)

    if args.save_json:
        out_path = Path(args.save_json)
        out_path.write_text(df.to_json(orient="records", force_ascii=False, indent=2), encoding="utf-8")
        print(f"\n✓ Results exported to {out_path}")


if __name__ == "__main__":
    main(sys.argv[1:])

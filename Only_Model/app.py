"""
Toxic Comment Classification API
Flask REST API for real-time toxic comment detection
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import re
import numpy as np
from scipy.sparse import hstack
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend integration

# ===========================
# Global Variables & Patterns
# ===========================

# Profanity patterns
PROFANITY_PATTERNS = [
    (r'f[\W_]*u[\W_]*c[\W_]*k', 'fuck'),
    (r'sh[\W_]*i[\W_]*t', 'shit'),
    (r'b[\W_]*i[\W_]*t[\W_]*c[\W_]*h', 'bitch'),
    (r'a[\W_]*s[\W_]*s[\W_]*h?[\W_]*o?[\W_]*l[\W_]*e?', 'asshole'),
    (r'd[\W_]*a[\W_]*m[\W_]*n', 'damn'),
    (r'h[\W_]*e[\W_]*l[\W_]*l', 'hell'),
    (r'idi0t', 'idiot'),
    (r'st\*pid', 'stupid'),
]

# Chat lingo normalization
CHAT_MAP = {
    r'\bu\b': 'you',
    r'\bur\b': 'your',
    r'\br\b': 'are',
}

# Positive words for context-aware profanity normalization
POSITIVE_WORDS = [
    "good", "great", "awesome", "amazing", "nice",
    "cool", "fun", "funny", "love", "lovely", "beautiful",
    "perfect", "excellent", "fantastic", "wonderful", "brilliant",
    "superb", "outstanding", "impressive", "incredible", "fabulous",
    "terrific", "magnificent", "marvelous", "spectacular", "phenomenal",
    "cute", "sweet", "adorable", "delightful", "charming",
    "interesting", "exciting", "thrilling", "enjoyable", "pleasant",
    "happy", "glad", "joyful", "pleased", "satisfied",
    "best", "better", "top", "fine", "solid", "strong",
    "smart", "clever", "genius", "wise", "talented"
]

# Create regex patterns
positive_pattern = "|".join(POSITIVE_WORDS)
BENIGN_PROFANITY_PATTERN = re.compile(
    rf"\b(fucking|fuckin|fking|freaking)\s+({positive_pattern})\b",
    flags=re.IGNORECASE
)
INTENSIFIED_PATTERN = re.compile(
    rf"\b(so|really|very|pretty|quite)\s+(fucking|fuckin|fking)\s+({positive_pattern})\b",
    flags=re.IGNORECASE
)

# Label columns
LABEL_COLS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

# Model variables (loaded on startup)
lr_models = None
tfidf_word = None
tfidf_char = None
lemmatizer = None


# ===========================
# Preprocessing Functions
# ===========================

def normalize_for_toxic(text):
    """
    Normalize text for toxic detection
    """
    # Lowercase
    text = text.lower()
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    
    # Replace benign profanity with intensifiers
    text = INTENSIFIED_PATTERN.sub(lambda m: f"{m.group(1)} very {m.group(3)}", text)
    text = BENIGN_PROFANITY_PATTERN.sub(lambda m: f"very {m.group(2)}", text)
    
    # Normalize leet speak: @ → a
    text = re.sub(r'@', 'a', text)
    
    # Collapse repeated characters
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    
    # Collapse repeated punctuation
    text = re.sub(r'!{2,}', '!', text)
    text = re.sub(r'\?{2,}', '?', text)
    text = re.sub(r'\.{2,}', '.', text)
    
    # Normalize obfuscated profanity
    for pattern, repl in PROFANITY_PATTERNS:
        text = re.sub(pattern, repl, text, flags=re.IGNORECASE)
    
    # Normalize chat lingo
    for pattern, repl in CHAT_MAP.items():
        text = re.sub(pattern, repl, text)
    
    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def analyzer_tfidf(text):
    """
    Custom analyzer for TF-IDF
    """
    try:
        tokens = word_tokenize(text)
        tokens = [lemmatizer.lemmatize(tok) for tok in tokens if tok.strip()]
        return tokens
    except:
        return text.split()


def predict_toxicity(text):
    """
    Predict toxicity for given text
    """
    # Normalize text
    normalized = normalize_for_toxic(text)
    
    # Vectorize
    vec_word = tfidf_word.transform([normalized])
    vec_char = tfidf_char.transform([normalized])
    vec = hstack([vec_word, vec_char])
    
    # Predict
    predictions = {}
    for label in LABEL_COLS:
        prob = lr_models[label].predict_proba(vec)[0, 1]
        predictions[label] = float(prob)
    
    return predictions, normalized


# ===========================
# API Endpoints
# ===========================

@app.route('/', methods=['GET'])
def home():
    """
    Home endpoint - API documentation
    """
    return jsonify({
        "name": "Toxic Comment Classification API",
        "version": "1.0.0",
        "description": "Real-time toxic comment detection using TF-IDF + Logistic Regression",
        "endpoints": {
            "GET /": "API documentation",
            "GET /health": "Health check",
            "POST /predict": "Predict toxicity for a comment",
            "POST /batch": "Predict toxicity for multiple comments"
        },
        "model_info": {
            "algorithm": "TF-IDF + Logistic Regression",
            "features": "Word n-grams (1-3) + Char n-grams (3-5)",
            "labels": LABEL_COLS,
            "preprocessing": [
                "Profanity normalization",
                "Context-aware profanity detection",
                "Leet speak normalization",
                "Chat lingo expansion"
            ]
        }
    })


@app.route('/health', methods=['GET'])
def health():
    """
    Health check endpoint
    """
    return jsonify({
        "status": "healthy",
        "models_loaded": lr_models is not None,
        "vectorizers_loaded": tfidf_word is not None and tfidf_char is not None
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict toxicity for a single comment
    
    Request body:
    {
        "text": "Your comment here",
        "threshold": 0.5  (optional, default: 0.5)
    }
    
    Response:
    {
        "text": "Original text",
        "normalized_text": "Preprocessed text",
        "predictions": {
            "toxic": 0.123,
            "severe_toxic": 0.045,
            ...
        },
        "is_toxic": true/false,
        "toxic_labels": ["toxic", "insult"],
        "max_toxicity": {
            "label": "toxic",
            "score": 0.856
        }
    }
    """
    try:
        # Get request data
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({
                "error": "Missing 'text' field in request body"
            }), 400
        
        text = data['text']
        threshold = data.get('threshold', 0.5)
        
        # Validate input
        if not isinstance(text, str) or len(text.strip()) == 0:
            return jsonify({
                "error": "Text must be a non-empty string"
            }), 400
        
        if not 0 <= threshold <= 1:
            return jsonify({
                "error": "Threshold must be between 0 and 1"
            }), 400
        
        # Predict
        predictions, normalized = predict_toxicity(text)
        
        # Determine toxicity
        toxic_labels = [label for label, prob in predictions.items() if prob > threshold]
        is_toxic = len(toxic_labels) > 0
        
        # Find max toxicity
        max_label = max(predictions.items(), key=lambda x: x[1])
        
        return jsonify({
            "text": text,
            "normalized_text": normalized,
            "predictions": predictions,
            "is_toxic": is_toxic,
            "toxic_labels": toxic_labels,
            "max_toxicity": {
                "label": max_label[0],
                "score": max_label[1]
            },
            "threshold": threshold
        })
        
    except Exception as e:
        return jsonify({
            "error": f"Prediction failed: {str(e)}"
        }), 500


@app.route('/batch', methods=['POST'])
def batch_predict():
    """
    Predict toxicity for multiple comments
    
    Request body:
    {
        "texts": ["comment 1", "comment 2", ...],
        "threshold": 0.5  (optional, default: 0.5)
    }
    
    Response:
    {
        "results": [
            { ... prediction for comment 1 ... },
            { ... prediction for comment 2 ... },
            ...
        ],
        "summary": {
            "total": 10,
            "toxic": 3,
            "clean": 7,
            "toxic_percentage": 30.0
        }
    }
    """
    try:
        # Get request data
        data = request.get_json()
        
        if not data or 'texts' not in data:
            return jsonify({
                "error": "Missing 'texts' field in request body"
            }), 400
        
        texts = data['texts']
        threshold = data.get('threshold', 0.5)
        
        # Validate input
        if not isinstance(texts, list):
            return jsonify({
                "error": "texts must be a list of strings"
            }), 400
        
        if len(texts) == 0:
            return jsonify({
                "error": "texts list cannot be empty"
            }), 400
        
        if len(texts) > 100:
            return jsonify({
                "error": "Maximum 100 texts per batch request"
            }), 400
        
        # Predict for each text
        results = []
        toxic_count = 0
        
        for text in texts:
            if not isinstance(text, str):
                results.append({
                    "error": "Invalid text (must be string)",
                    "text": str(text)
                })
                continue
            
            predictions, normalized = predict_toxicity(text)
            toxic_labels = [label for label, prob in predictions.items() if prob > threshold]
            is_toxic = len(toxic_labels) > 0
            
            if is_toxic:
                toxic_count += 1
            
            max_label = max(predictions.items(), key=lambda x: x[1])
            
            results.append({
                "text": text,
                "normalized_text": normalized,
                "predictions": predictions,
                "is_toxic": is_toxic,
                "toxic_labels": toxic_labels,
                "max_toxicity": {
                    "label": max_label[0],
                    "score": max_label[1]
                }
            })
        
        return jsonify({
            "results": results,
            "summary": {
                "total": len(texts),
                "toxic": toxic_count,
                "clean": len(texts) - toxic_count,
                "toxic_percentage": round(toxic_count / len(texts) * 100, 2)
            },
            "threshold": threshold
        })
        
    except Exception as e:
        return jsonify({
            "error": f"Batch prediction failed: {str(e)}"
        }), 500


# ===========================
# Model Loading
# ===========================

def load_models():
    """
    Load trained models and vectorizers
    """
    global lr_models, tfidf_word, tfidf_char, lemmatizer
    
    print("Loading models...")
    
    try:
        # Load models
        with open('models/lr_models.pkl', 'rb') as f:
            lr_models = pickle.load(f)
        
        # Load vectorizers
        with open('models/tfidf_word.pkl', 'rb') as f:
            tfidf_word = pickle.load(f)
        
        with open('models/tfidf_char.pkl', 'rb') as f:
            tfidf_char = pickle.load(f)
        
        # Initialize lemmatizer
        lemmatizer = WordNetLemmatizer()
        
        print("✓ Models loaded successfully!")
        return True
        
    except FileNotFoundError:
        print("❌ Error: Model files not found!")
        print("Please run 'save_models.py' first to save trained models.")
        return False
    except Exception as e:
        print(f"❌ Error loading models: {str(e)}")
        return False


# ===========================
# Main Entry Point
# ===========================

if __name__ == '__main__':
    # Load models on startup
    if not load_models():
        print("\n⚠️  API cannot start without models!")
        print("Run these steps:")
        print("1. Train your model in test_notebook.ipynb")
        print("2. Run 'python save_models.py' to export models")
        print("3. Run 'python app.py' to start API")
        exit(1)
    
    # Start Flask app
    print("\n" + "="*50)
    print("🚀 Starting Toxic Comment Classification API")
    print("="*50)
    print("API running at: http://localhost:5000")
    print("Documentation: http://localhost:5000")
    print("Health check: http://localhost:5000/health")
    print("\nPress CTRL+C to stop")
    print("="*50 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=True)

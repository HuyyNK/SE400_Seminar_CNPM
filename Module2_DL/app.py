"""
FastAPI Server for Module 2 - Toxic Comment Classifier with Optimal Thresholds

Usage:
    uvicorn app:app --reload --port 8000

Endpoints:
    GET  / - Web UI for testing
    GET  /health - Health check
    POST /predict - Single text prediction with optimal thresholds
    POST /predict/batch - Batch predictions
    GET  /thresholds - Get optimal thresholds
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, RootModel
from typing import List, Dict, Optional
import sys
import os
import json
import numpy as np

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils import PROJECT_ROOT, LABEL_COLS as LABEL_COLUMNS
from src.processing import PostProcessingPipeline
from src.preprocessing import TextPreprocessor, prepare_char_sequences, get_default_char_vocab
from src.utils import DEFAULT_CHAR_VOCAB as CORRECT_CHAR_VOCAB

# Lazy imports for TensorFlow (to avoid import issues)
keras = None
tokenizer_from_json = None

# Initialize FastAPI
app = FastAPI(
    title="Toxic Comment Classifier API",
    description="Deep Learning API with Optimal Thresholds (Macro F1: 0.5495)",
    version="2.1"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
MODEL = None
TOKENIZER = None
PREPROCESSOR = None
POST_PROCESSOR = None
OPTIMAL_THRESHOLDS = {}
CONFIG = {}

def load_model_artifacts():
    """Load model, tokenizer, and thresholds at startup"""
    global MODEL, TOKENIZER, PREPROCESSOR, OPTIMAL_THRESHOLDS, CONFIG
    
    # Import TensorFlow here to avoid import issues
    from tensorflow import keras
    from tensorflow.keras.preprocessing.text import tokenizer_from_json
    
    try:
        from src.core import ToxicCommentClassifier, load_glove_embeddings
        import string
        
        artifacts_dir = PROJECT_ROOT / "artifacts"
        
        # Load tokenizer first
        tokenizer_path = artifacts_dir / "tokenizer.json"
        with open(tokenizer_path, 'r', encoding='utf-8') as f:
            TOKENIZER = tokenizer_from_json(f.read())
        print(f"[OK] Tokenizer loaded: {len(TOKENIZER.word_index)} words")
        
        # Try to load cached embedding matrix (fast) or generate from GloVe (slow)
        embedding_cache = artifacts_dir / "embedding_matrix.npy"
        
        if embedding_cache.exists():
            # Load from cache (1-2 seconds)
            print("Loading GloVe embeddings from cache...")
            embedding_matrix = np.load(str(embedding_cache))
            print(f"[OK] GloVe embeddings loaded from cache: {embedding_matrix.shape}")
        else:
            # Load GloVe and cache it (1-2 minutes first time)
            print("Loading GloVe embeddings from file (first time - will cache)...")
            glove_path = PROJECT_ROOT / "embeddings" / "glove.6B.300d.txt"
            embedding_matrix = load_glove_embeddings(
                word_index=TOKENIZER.word_index,
                glove_path=str(glove_path),
                embedding_dim=300,
                max_words=50000
            )
            print(f"[OK] GloVe embeddings loaded: {embedding_matrix.shape}")
            
            # Cache for next time
            print("Caching embedding matrix for faster startup next time...")
            np.save(str(embedding_cache), embedding_matrix)
            print(f"[OK] Cache saved: {embedding_cache}")

        
        # Rebuild model architecture
        print("Building model architecture...")
        
        classifier = ToxicCommentClassifier(
            vocab_size=50000,
            embedding_dim=300,
            max_len=150,
            char_vocab_size=len(CORRECT_CHAR_VOCAB),
            max_char_len=200,
            lstm_units_layer1=128,
            lstm_units_layer2=64,
            lstm_units_layer3=32,
            char_emb_dim=48,
            char_num_filters=128,
            num_attention_heads=2,
            attention_key_dim=16,
            trainable_embedding=False,
            l2_reg=0.01
        )
        
        MODEL = classifier.build(embedding_matrix=embedding_matrix)
        print(f"[OK] Model built: {MODEL.count_params():,} parameters")
        
        # Load pre-trained weights
        model_path = artifacts_dir / "models" / "best_model.h5"
        print(f"Loading weights from {model_path}...")
        try:
            # Load weights only (avoid Lambda layer deserialization)
            MODEL.load_weights(str(model_path))
            print("[OK] Weights loaded successfully")
        except Exception as e:
            print(f"⚠ Warning: Could not load weights: {e}")
            print("  Model will use random initialization")
        
        # Load config with defaults
        config_path = artifacts_dir / "config.json"
        CONFIG = {
            'max_len': 150,
            'max_char_len': 200,
            'embedding_dim': 300,
            'char_emb_dim': 48,
            'char_num_filters': 128
        }
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    loaded_config = json.load(f)
                    CONFIG.update(loaded_config)
                print("[OK] Config loaded from file")
            except Exception as e:
                print(f"[WARNING] Could not load config file, using defaults: {e}")
        else:
            print("[OK] Using default config values")
        
        # Load optimal thresholds
        thresholds_path = artifacts_dir / "reports" / "optimized_thresholds.json"
        if thresholds_path.exists():
            with open(thresholds_path, 'r') as f:
                data = json.load(f)
                OPTIMAL_THRESHOLDS = data.get('optimal_thresholds', {})
            print(f"[OK] Optimal thresholds loaded: {OPTIMAL_THRESHOLDS}")
        else:
            # Default to 0.5 if file not found
            OPTIMAL_THRESHOLDS = {label: 0.5 for label in LABEL_COLUMNS}
            print(f"⚠ Using default thresholds: 0.5")
        
        # Initialize preprocessor
        PREPROCESSOR = TextPreprocessor()
        print("[OK] Preprocessor initialized")
        
        # Initialize post-processing pipeline
        global POST_PROCESSOR
        POST_PROCESSOR = PostProcessingPipeline()
        print("[OK] Post-processing pipeline initialized")
        
        print("\n[SUCCESS] All artifacts loaded successfully!")
        return True
    except Exception as e:
        import traceback
        print(f"[ERROR] Failed to load artifacts: {e}")
        traceback.print_exc()
        return False

# Load at startup
load_model_artifacts()


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def prepare_sequences(texts, tokenizer, max_len):
    """Convert texts to word sequences"""
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')
    return padded


def get_default_char_vocab():
    """Get default character vocabulary"""
    return CORRECT_CHAR_VOCAB


# Request/Response models
class SingleTextRequest(BaseModel):
    text: str

class LabelPrediction(BaseModel):
    probability: float
    predicted: bool
    threshold_used: float

class PredictionResponse(BaseModel):
    text: str
    predictions: Dict[str, LabelPrediction]
    toxic_labels: List[str]
    is_toxic: bool
    risk_level: str

class MultipleTextsRequest(BaseModel):
    texts: List[str]

class BatchPredictionResponse(BaseModel):
    results: List[PredictionResponse]
    summary: Dict[str, int]

class SimpleBatchItem(BaseModel):
    text: str
    is_toxic: bool

class SimpleBatchResponse(RootModel[List[SimpleBatchItem]]):
    root: List[SimpleBatchItem]


@app.get("/", response_class=HTMLResponse)
def root():
    """Web UI for testing"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Toxic Comment Classifier</title>
        <meta charset="UTF-8">
        <style>
            body { font-family: Arial, sans-serif; max-width: 900px; margin: 50px auto; padding: 20px; background: #f5f5f5; }
            .container { background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            h1 { color: #333; margin-bottom: 10px; }
            .subtitle { color: #666; margin-bottom: 30px; }
            textarea { width: 100%; height: 150px; padding: 15px; font-size: 14px; border: 2px solid #ddd; border-radius: 5px; box-sizing: border-box; font-family: inherit; }
            button { background: #007bff; color: white; padding: 12px 30px; border: none; cursor: pointer; font-size: 16px; margin-top: 15px; border-radius: 5px; }
            button:hover { background: #0056b3; }
            #result { margin-top: 30px; padding: 20px; border: 2px solid #ddd; background: #fafafa; border-radius: 5px; display: none; }
            .status { font-size: 24px; font-weight: bold; margin-bottom: 15px; }
            .toxic { color: #dc3545; }
            .safe { color: #28a745; }
            .labels-container { margin-top: 20px; }
            .label { display: inline-block; margin: 5px; padding: 8px 15px; border-radius: 20px; font-size: 13px; }
            .label.positive { background: #ffe0e0; border: 2px solid #ff4444; color: #cc0000; font-weight: bold; }
            .label.negative { background: #e8e8e8; border: 2px solid #ccc; color: #666; }
            .prob { font-size: 11px; opacity: 0.8; }
            .example-buttons { margin-top: 15px; }
            .example-btn { background: #6c757d; margin-right: 10px; padding: 8px 15px; font-size: 13px; }
            .example-btn:hover { background: #5a6268; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🛡️ Toxic Comment Classifier</h1>
            <p class="subtitle">Module 2 - Deep Learning with Optimal Thresholds | Macro F1: 0.5495 | AUC: 0.9827</p>
            
            <textarea id="textInput" placeholder="Nhập text để kiểm tra toxic detection..."></textarea>
            
            <div class="example-buttons">
                <button class="example-btn" onclick="setExample('This is a great article!')">Safe Example</button>
                <button class="example-btn" onclick="setExample('You are stupid and worthless!')">Toxic Example</button>
                <button class="example-btn" onclick="setExample('I will find you and hurt you badly')">Threat Example</button>
            </div>
            
            <br>
            <button onclick="predict()">🔍 Kiểm tra Toxic</button>
            
            <div id="result"></div>
        </div>
        
        <script>
        function setExample(text) {
            document.getElementById('textInput').value = text;
        }
        
        async function predict() {
            const text = document.getElementById('textInput').value;
            if (!text.trim()) {
                alert('Vui lòng nhập text!');
                return;
            }
            
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({text: text})
                });
                
                if (!response.ok) {
                    throw new Error('Server error: ' + response.statusText);
                }
                
                const data = await response.json();
                
                let html = '<div class="status ' + (data.is_toxic ? 'toxic' : 'safe') + '">';
                html += data.is_toxic ? '⚠️ TOXIC DETECTED' : '✅ SAFE';
                html += ' (' + data.risk_level + ')</div>';
                
                if (data.toxic_labels.length > 0) {
                    html += '<p><strong>Loại toxic phát hiện:</strong> ' + data.toxic_labels.join(', ') + '</p>';
                }
                
                html += '<div class="labels-container"><h4>Chi tiết từng label:</h4>';
                for (const [label, pred] of Object.entries(data.predictions)) {
                    const pct = (pred.probability * 100).toFixed(1);
                    const cls = pred.predicted ? 'positive' : 'negative';
                    html += '<div class="label ' + cls + '">';
                    html += '<strong>' + label + '</strong>: ' + pct + '%';
                    html += ' <span class="prob">(threshold: ' + pred.threshold_used.toFixed(2) + ')</span>';
                    html += '</div>';
                }
                html += '</div>';
                
                document.getElementById('result').innerHTML = html;
                document.getElementById('result').style.display = 'block';
            } catch (error) {
                alert('Lỗi: ' + error.message);
                console.error(error);
            }
        }
        </script>
    </body>
    </html>
    """
    return html_content


@app.get("/health")
def health():
    """Health check endpoint"""
    return {
        "status": "running" if MODEL is not None else "error",
        "model": "Module 2 - Toxic Comment Classifier",
        "version": "2.1",
        "labels": LABEL_COLUMNS,
        "optimal_thresholds": OPTIMAL_THRESHOLDS,
        "metrics": {
            "macro_f1": 0.5495,
            "macro_auc": 0.9827,
            "macro_recall": 0.7075
        }
    }


@app.get("/thresholds")
def get_thresholds():
    """Get optimal thresholds"""
    return {
        "optimal_thresholds": OPTIMAL_THRESHOLDS,
        "description": "F1-optimized thresholds per label",
        "macro_f1": 0.5495,
        "improvement": "56.6% over default threshold=0.5"
    }


@app.post("/predict", response_model=PredictionResponse)
def predict_single(request: SingleTextRequest):
    """
    API 1: Predict toxicity for a single text
    
    Args:
        text: Input text to classify
    
    Returns:
        Detailed predictions with probabilities, thresholds, and risk level
    """
    if MODEL is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    try:
        # Get config values
        max_len = CONFIG.get('max_len', 150)
        max_char_len = CONFIG.get('max_char_len', 200)
        
        # Preprocess
        clean_text = PREPROCESSOR.preprocess(request.text)
        
        # Tokenize
        X_word = prepare_sequences([clean_text], TOKENIZER, max_len)
        
        # Create character mapping with CORRECT_CHAR_VOCAB
        char_mapping = {c: i+1 for i, c in enumerate(CORRECT_CHAR_VOCAB)}
        X_char = prepare_char_sequences([clean_text], max_char_len, char_mapping)
        
        # Predict
        probas = MODEL.predict([X_word, X_char], verbose=0)[0]
        
        # Use optimal thresholds
        thresholds = OPTIMAL_THRESHOLDS
        
        # Build detailed predictions
        predictions = {}
        toxic_labels = []
        max_prob = 0
        
        for label, proba in zip(LABEL_COLUMNS, probas):
            threshold = thresholds.get(label, 0.5)
            predicted = float(proba) >= threshold
            
            predictions[label] = {
                "probability": float(proba),
                "predicted": predicted,
                "threshold_used": threshold
            }
            
            if predicted:
                toxic_labels.append(label)
            
            max_prob = max(max_prob, float(proba))
        
        is_toxic = len(toxic_labels) > 0
        
        # Determine risk level
        if not is_toxic:
            risk_level = "Safe"
        elif max_prob >= 0.9:
            risk_level = "High Risk"
        elif max_prob >= 0.7:
            risk_level = "Medium Risk"
        else:
            risk_level = "Low Risk"
        
        # 🆕 Apply post-processing filters
        predictions_dict = {
            label: {
                "probability": float(proba),
                "predicted": label in toxic_labels,
                "threshold_used": OPTIMAL_THRESHOLDS.get(label, 0.5)
            }
            for label, proba in zip(LABEL_COLUMNS, probas)
        }
        
        # Apply post-processing filters (identity_hate filter, threat boosting, etc.)
        # This modifies predictions_dict and adds meta keys
        predictions_dict = POST_PROCESSOR.apply(request.text, predictions_dict)
        
        # Extract meta keys added by post-processing
        toxic_labels = predictions_dict.pop("toxic_labels", [])
        is_toxic = predictions_dict.pop("is_toxic", False)
        risk_level = predictions_dict.pop("risk_level", risk_level)
        predictions_dict.pop("metadata", None)  # Remove metadata key if present
        
        response_dict = {
            "text": request.text[:100] + "..." if len(request.text) > 100 else request.text,
            "predictions": predictions_dict,
            "toxic_labels": toxic_labels,
            "is_toxic": is_toxic,
            "risk_level": risk_level
        }
        
        # Convert back to Pydantic model
        return PredictionResponse(**response_dict)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch", response_model=SimpleBatchResponse)
def predict_batch(request: MultipleTextsRequest):
    """
    API 2: Batch prediction for multiple texts (simplified)
    
    Args:
        texts: List of texts to classify
    
    Returns:
        List of {text, is_toxic} for each text
    """
    if MODEL is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    if not request.texts:
        raise HTTPException(status_code=400, detail="Texts list cannot be empty")
    
    try:
        # Use optimal thresholds
        thresholds = OPTIMAL_THRESHOLDS
        
        results = []
        
        # Get config values
        max_len = CONFIG.get('max_len', 150)
        max_char_len = CONFIG.get('max_char_len', 200)
        
        for text in request.texts:
            if not text or not text.strip():
                continue
            
            # Preprocess
            clean_text = PREPROCESSOR.preprocess(text)
            
            # Tokenize
            X_word = prepare_sequences([clean_text], TOKENIZER, max_len)
            
            # Create character mapping with CORRECT_CHAR_VOCAB
            char_mapping = {c: i+1 for i, c in enumerate(CORRECT_CHAR_VOCAB)}
            X_char = prepare_char_sequences([clean_text], max_char_len, char_mapping)
            
            # Predict
            probas = MODEL.predict([X_word, X_char], verbose=0)[0]
            
            # Check if toxic (any label exceeds threshold)
            toxic_labels = []
            for label, proba in zip(LABEL_COLUMNS, probas):
                threshold = thresholds.get(label, 0.5)
                if float(proba) >= threshold:
                    toxic_labels.append(label)
            
            is_toxic = len(toxic_labels) > 0
            
            results.append(SimpleBatchItem(
                text=text,
                is_toxic=is_toxic
            ))
        
        return SimpleBatchResponse(root=results)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*60)
    print("Starting Toxic Comment Classifier API")
    print("="*60)
    print("Open http://localhost:8000 in your browser")
    print("="*60 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)


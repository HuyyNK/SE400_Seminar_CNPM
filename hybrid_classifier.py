# hybrid_classifier.py

import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Text cleaning function
def clean_text(text):
    """
    Clean and preprocess text data
    """
    # Convert to string and lowercase
    text = str(text).lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove user mentions and hashtags
    text = re.sub(r'@\w+|#\w+', '', text)
    
    # Remove HTML entities
    text = re.sub(r'&\w+;|&#\d+;', '', text)
    
    # Remove special characters and numbers (keep only letters and spaces)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    
    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# Advanced preprocessing with lemmatization
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """
    Advanced preprocessing: tokenization, stopword removal, lemmatization
    """
    try:
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        tokens = [lemmatizer.lemmatize(word) for word in tokens 
                  if word not in stop_words and len(word) > 2]
        
        return ' '.join(tokens)
    except:
        return text

# Hybrid Prediction Class
class HybridToxicClassifier:
    """
    Hybrid classifier combining Rule-based Filter + ML Model
    """
    
    def __init__(self, ml_model, vectorizer, rule_detector=None, ml_threshold=0.5):
        self.ml_model = ml_model
        self.vectorizer = vectorizer
        self.rule_detector = rule_detector
        self.ml_threshold = ml_threshold
    
    def predict(self, text, return_details=False):
        """
        Predict using hybrid approach
        
        Args:
            text: Input text
            return_details: If True, return detailed prediction info
        
        Returns:
            dict with prediction results
        """
        # Step 1: Rule-based filter
        rule_violation = False
        rule_phrases = []
        rule_confidence = 0.0
        
        if self.rule_detector is not None:
            try:
                rule_result = self.rule_detector.detect(text, return_details=True)
                rule_violation = rule_result.get('is_toxic', False)
                rule_phrases = rule_result.get('toxic_phrases', [])
                
                if rule_violation:
                    # High confidence for rule-based detection
                    return {
                        'text': text,
                        'is_violation': True,
                        'label': 'VIOLATION',
                        'method': 'rule_based',
                        'ml_probability': None,
                        'confidence': 0.95,
                        'toxic_phrases': rule_phrases,
                        'details': 'Detected by rule-based filter'
                    }
            except Exception as e:
                print(f"Rule detector error: {e}")
        
        # Step 2: ML Model prediction
        # Preprocess text
        cleaned = clean_text(text)
        processed = preprocess_text(cleaned)
        
        # Vectorize
        vectorized = self.vectorizer.transform([processed])
        
        # Predict
        ml_prediction = self.ml_model.predict(vectorized)[0]
        ml_probability = self.ml_model.predict_proba(vectorized)[0][1]
        
        # Apply threshold
        is_violation = ml_probability >= self.ml_threshold
        
        return {
            'text': text,
            'is_violation': bool(is_violation),
            'label': 'VIOLATION' if is_violation else 'SAFE',
            'method': 'ml_model',
            'ml_probability': float(ml_probability),
            'confidence': float(ml_probability) if is_violation else float(1 - ml_probability),
            'toxic_phrases': rule_phrases,
            'details': f'ML probability: {ml_probability:.4f}'
        }

print("✓ HybridToxicClassifier defined")
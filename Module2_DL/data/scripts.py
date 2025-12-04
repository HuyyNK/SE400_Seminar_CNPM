"""
Module 2 DL - All Scripts Consolidated
======================================

This file consolidates all utility scripts from scripts/ folder:
1. Data combination (combine.py)
2. Model evaluation (evaluate.py)
3. Validation (validate_phase.py)
4. Threshold optimization (find_thresholds.py)
5. Artifact export (export_artifacts.py)

Usage:
    python scripts.py combine --help
    python scripts.py evaluate --help
    python scripts.py validate --help
    python scripts.py find_thresholds --help
    python scripts.py export_artifacts --help
"""

import sys
import os
import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, roc_auc_score, f1_score,
    precision_score, recall_score, hamming_loss, accuracy_score
)

# Add parent directory (Module2_DL) to path so we can import src
module2_root = Path(__file__).parent.parent
sys.path.insert(0, str(module2_root))

# Conditional imports (will be imported when needed)
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.preprocessing.text import Tokenizer, tokenizer_from_json
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


# ============================================================================
# CONSTANTS
# ============================================================================

LABEL_COLS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
MAX_WORDS = 50000
MAX_LEN = 250
MAX_CHAR_LEN = 400


# ============================================================================
# 1. DATA COMBINATION
# ============================================================================

def cmd_combine(args):
    """Combine Kaggle, Twitter, and augmented datasets"""
    from src.utils import (
        KAGGLE_TRAIN_CSV, TWITTER_TRAIN_CSV, get_artifact_path,
        PROJECT_ROOT
    )
    
    # Define augmented data paths
    PHASE2_AUGMENTED_CSV = PROJECT_ROOT / 'Data' / 'phase2_augmented.csv'
    OBFUSCATION_AUGMENTED_CSV = PROJECT_ROOT / 'Data' / 'obfuscation_augmented.csv'
    THREAT_AUGMENTED_CSV = PROJECT_ROOT / 'Data' / 'threat_augmented.csv'
    TRAIN_COMBINED_CSV = get_artifact_path('train_combined.csv')
    TRAIN_FINAL_CSV = get_artifact_path('train_final.csv')
    VAL_FINAL_CSV = get_artifact_path('val_final.csv')
    
    print("="*80)
    print("DATA COMBINATION")
    print("="*80)
    
    # Load datasets
    print("\n[1/6] Loading datasets...")
    datasets = []
    
    # Kaggle
    try:
        df_kaggle = pd.read_csv(KAGGLE_TRAIN_CSV)
        print(f"  ✓ Kaggle: {len(df_kaggle):,} samples")
        datasets.append(('kaggle', df_kaggle))
    except FileNotFoundError:
        print(f"  ✗ Kaggle data not found")
    
    # Twitter
    try:
        df_twitter = pd.read_csv(TWITTER_TRAIN_CSV)
        print(f"  ✓ Twitter: {len(df_twitter):,} samples")
        
        # Standardize Twitter format
        if 'class' in df_twitter.columns:
            df_twitter['toxic'] = (df_twitter['class'] > 0).astype(int)
        if 'tweet' in df_twitter.columns:
            df_twitter = df_twitter.rename(columns={'tweet': 'comment_text'})
        
        for col in LABEL_COLS:
            if col not in df_twitter.columns:
                df_twitter[col] = 0
        
        datasets.append(('twitter', df_twitter))
    except FileNotFoundError:
        print(f"  ✗ Twitter data not found")
    
    # Phase 2 augmented
    try:
        df_phase2 = pd.read_csv(PHASE2_AUGMENTED_CSV)
        print(f"  ✓ Phase 2 augmented: {len(df_phase2):,} samples")
        datasets.append(('phase2', df_phase2))
    except FileNotFoundError:
        print(f"  ✗ Phase 2 augmented not found")
    
    # Obfuscation augmented
    try:
        df_obfuscation = pd.read_csv(OBFUSCATION_AUGMENTED_CSV)
        print(f"  ✓ Obfuscation: {len(df_obfuscation):,} samples")
        datasets.append(('obfuscation', df_obfuscation))
    except FileNotFoundError:
        print(f"  ✗ Obfuscation augmented not found")
    
    # Threat augmented
    try:
        df_threats = pd.read_csv(THREAT_AUGMENTED_CSV)
        print(f"  ✓ Threats: {len(df_threats):,} samples")
        datasets.append(('threats', df_threats))
    except FileNotFoundError:
        print(f"  ✗ Threat augmented not found")
    
    if not datasets:
        print("\n❌ No datasets found!")
        return
    
    # Standardize columns
    print("\n[2/6] Standardizing columns...")
    processed = []
    
    for name, df in datasets:
        for col in LABEL_COLS:
            if col not in df.columns:
                df[col] = 0
        
        keep_cols = ['comment_text'] + LABEL_COLS
        df = df[keep_cols].copy()
        processed.append(df)
        print(f"  ✓ {name}: {len(df):,} samples")
    
    # Combine
    print("\n[3/6] Combining datasets...")
    df_combined = pd.concat(processed, ignore_index=True)
    
    # Remove duplicates
    original_len = len(df_combined)
    df_combined = df_combined.drop_duplicates(subset=['comment_text'], keep='first')
    duplicates_removed = original_len - len(df_combined)
    
    print(f"  ✓ Combined: {len(df_combined):,} samples")
    print(f"  ✓ Removed {duplicates_removed:,} duplicates")
    
    # Shuffle
    df_combined = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Label distribution
    print("\n[4/6] Label distribution:")
    for label in LABEL_COLS:
        count = df_combined[label].sum()
        pct = count / len(df_combined) * 100
        print(f"  {label:15s}: {count:6,} ({pct:5.2f}%)")
    
    # Train/val split
    val_split = args.val_split if hasattr(args, 'val_split') else 0.15
    print(f"\n[5/6] Splitting ({int((1-val_split)*100)}% / {int(val_split*100)}%)...")
    
    train_df, val_df = train_test_split(
        df_combined,
        test_size=val_split,
        random_state=42,
        stratify=df_combined['toxic']
    )
    
    print(f"  ✓ Train: {len(train_df):,} samples")
    print(f"  ✓ Val:   {len(val_df):,} samples")
    
    # Save
    print(f"\n[6/6] Saving files...")
    TRAIN_COMBINED_CSV.parent.mkdir(parents=True, exist_ok=True)
    
    df_combined.to_csv(TRAIN_COMBINED_CSV, index=False)
    train_df.to_csv(TRAIN_FINAL_CSV, index=False)
    val_df.to_csv(VAL_FINAL_CSV, index=False)
    
    print(f"  ✓ Combined: {TRAIN_COMBINED_CSV}")
    print(f"  ✓ Train:    {TRAIN_FINAL_CSV}")
    print(f"  ✓ Val:      {VAL_FINAL_CSV}")
    
    print("\n" + "="*80)
    print("✅ DATA COMBINATION COMPLETE")
    print("="*80)


# ============================================================================
# 2. MODEL EVALUATION
# ============================================================================

def cmd_evaluate(args):
    """Evaluate trained model"""
    if not TF_AVAILABLE:
        print("❌ TensorFlow not installed. Run: pip install tensorflow")
        return
    
    from src.preprocessing import TextPreprocessor, prepare_sequences, prepare_char_sequences, get_default_char_vocab
    
    print("="*80)
    print("MODEL EVALUATION")
    print("="*80)
    print(f"Model: {args.model_path}")
    print(f"Test data: {args.test_csv}")
    print("="*80)
    
    # Load model
    print("\n[1/5] Loading model...")
    model = keras.models.load_model(args.model_path, compile=False)
    print("  ✓ Model loaded")
    
    # Load test data
    print("\n[2/5] Loading test data...")
    df_test = pd.read_csv(args.test_csv)
    X_text = df_test['comment_text'].astype(str).tolist()
    y_true = df_test[LABEL_COLS].values
    print(f"  ✓ Loaded {len(df_test):,} samples")
    
    # Preprocess
    print("\n[3/5] Preprocessing...")
    preprocessor = TextPreprocessor()
    X_text_clean = preprocessor.preprocess_batch(X_text)
    print(f"  ✓ Preprocessed")
    
    # Tokenize
    print("\n[4/5] Tokenizing...")
    
    # Load tokenizers
    artifacts_dir = Path(args.model_path).parent.parent / "artifacts"
    tokenizer_path = artifacts_dir / "tokenizer.json"
    
    if not tokenizer_path.exists():
        print(f"❌ Tokenizer not found: {tokenizer_path}")
        return
    
    with open(tokenizer_path, 'r', encoding='utf-8') as f:
        word_tokenizer = tokenizer_from_json(f.read())
    
    char_vocab = get_default_char_vocab()
    char_tokenizer = type('obj', (object,), {'word_index': {c: i+1 for i, c in enumerate(char_vocab)}})()
    
    X_word = prepare_sequences(X_text_clean, word_tokenizer, max_len=MAX_LEN)
    X_char = prepare_char_sequences(X_text_clean, MAX_CHAR_LEN, char_tokenizer.word_index)
    
    print(f"  ✓ Word sequences: {X_word.shape}")
    print(f"  ✓ Char sequences: {X_char.shape}")
    
    # Predict
    print("\n[5/5] Evaluating...")
    y_pred_proba = model.predict([X_word, X_char], batch_size=128, verbose=1)
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    # Metrics
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    
    # Overall
    hamming = hamming_loss(y_true, y_pred)
    subset_acc = accuracy_score(y_true, y_pred)
    
    print(f"\nOverall:")
    print(f"  Hamming Loss:    {hamming:.6f}")
    print(f"  Subset Accuracy: {subset_acc:.4f}")
    
    # Per-label
    print(f"\nPer-label:")
    for idx, label in enumerate(LABEL_COLS):
        auc = roc_auc_score(y_true[:, idx], y_pred_proba[:, idx])
        f1 = f1_score(y_true[:, idx], y_pred[:, idx])
        precision = precision_score(y_true[:, idx], y_pred[:, idx], zero_division=0)
        recall = recall_score(y_true[:, idx], y_pred[:, idx], zero_division=0)
        
        print(f"\n  {label}:")
        print(f"    AUC:       {auc:.4f}")
        print(f"    F1:        {f1:.4f}")
        print(f"    Precision: {precision:.4f}")
        print(f"    Recall:    {recall:.4f}")
    
    # Macro avg
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    macro_precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    macro_recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    
    print(f"\nMacro Average:")
    print(f"  F1:        {macro_f1:.4f}")
    print(f"  Precision: {macro_precision:.4f}")
    print(f"  Recall:    {macro_recall:.4f}")
    
    # Save results
    if args.output:
        results = {
            'timestamp': datetime.now().isoformat(),
            'hamming_loss': float(hamming),
            'subset_accuracy': float(subset_acc),
            'macro_f1': float(macro_f1),
            'macro_precision': float(macro_precision),
            'macro_recall': float(macro_recall)
        }
        
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✓ Results saved: {args.output}")
    
    print("\n" + "="*80)
    print("✅ EVALUATION COMPLETE")
    print("="*80)


# ============================================================================
# 3. VALIDATION
# ============================================================================

def cmd_validate(args):
    """Validate model improvements"""
    if not REQUESTS_AVAILABLE:
        print("❌ requests not installed. Run: pip install requests")
        return
    
    print("="*80)
    print("PHASE VALIDATION")
    print("="*80)
    print(f"API: {args.api_url}")
    print("="*80)
    
    # Check API
    try:
        response = requests.get(args.api_url.replace('/predict', '/health'), timeout=5)
        if response.status_code != 200:
            print(f"❌ API not responding")
            return
    except Exception as e:
        print(f"❌ Cannot connect to API: {e}")
        return
    
    # Test cases (simplified from validate_phase.py)
    test_cases = {
        "clean": [
            "You are a wonderful person!",
            "I disagree but respect your opinion.",
        ],
        "toxic": [
            "What the f**k are u talking about?",
            "B**ch please u dont know anything",
        ]
    }
    
    errors = 0
    total = 0
    
    print("\n[1/2] Testing clean detection...")
    for text in test_cases["clean"]:
        try:
            response = requests.post(args.api_url, json={"text": text}, timeout=10)
            pred = response.json()
            
            is_toxic = pred.get("is_toxic", False)
            if is_toxic:
                print(f"  ❌ False positive: {text[:50]}")
                errors += 1
            else:
                print(f"  ✓ Correct: {text[:50]}")
            total += 1
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    print("\n[2/2] Testing toxic detection...")
    for text in test_cases["toxic"]:
        try:
            response = requests.post(args.api_url, json={"text": text}, timeout=10)
            pred = response.json()
            
            is_toxic = pred.get("is_toxic", False)
            if not is_toxic:
                print(f"  ❌ False negative: {text[:50]}")
                errors += 1
            else:
                print(f"  ✓ Correct: {text[:50]}")
            total += 1
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    print("\n" + "="*80)
    print(f"RESULTS: {total - errors}/{total} correct ({(total-errors)/total*100:.1f}%)")
    print("="*80)


# ============================================================================
# 4. THRESHOLD OPTIMIZATION
# ============================================================================

def cmd_find_thresholds(args):
    """Find optimal thresholds for each label"""
    print("="*80)
    print("THRESHOLD OPTIMIZATION")
    print("="*80)
    print(f"Predictions: {args.predictions}")
    print(f"Labels: {args.labels}")
    print("="*80)
    
    # Load data
    print("\n[1/3] Loading data...")
    y_pred_proba = np.load(args.predictions)
    df = pd.read_csv(args.labels)
    y_true = df[LABEL_COLS].values
    
    print(f"  ✓ Loaded {len(y_true):,} samples")
    
    # Find thresholds
    print("\n[2/3] Finding optimal thresholds...")
    optimal_thresholds = {}
    
    for idx, label in enumerate(LABEL_COLS):
        best_f1 = 0
        best_threshold = 0.5
        
        for threshold in np.arange(0.1, 0.91, 0.05):
            y_pred = (y_pred_proba[:, idx] >= threshold).astype(int)
            f1 = f1_score(y_true[:, idx], y_pred, zero_division=0)
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        optimal_thresholds[label] = float(best_threshold)
        print(f"  {label:15s}: {best_threshold:.2f} (F1={best_f1:.4f})")
    
    # Save
    print("\n[3/3] Saving results...")
    output_path = Path(args.output) if args.output else Path("optimal_thresholds.json")
    
    with open(output_path, 'w') as f:
        json.dump({'thresholds': optimal_thresholds}, f, indent=2)
    
    print(f"  ✓ Saved: {output_path}")
    
    print("\n" + "="*80)
    print("✅ OPTIMIZATION COMPLETE")
    print("="*80)


# ============================================================================
# 5. EXPORT ARTIFACTS
# ============================================================================

def cmd_export_artifacts(args):
    """Export tokenizers and config"""
    if not TF_AVAILABLE:
        print("❌ TensorFlow not installed. Run: pip install tensorflow")
        return
    
    from src.preprocessing import TextPreprocessor
    
    print("="*80)
    print("EXPORT ARTIFACTS")
    print("="*80)
    print(f"Source: {args.source_csv}")
    print("="*80)
    
    # Load data
    print("\n[1/3] Loading data...")
    df = pd.read_csv(args.source_csv)
    texts = df['comment_text'].astype(str).tolist()
    print(f"  ✓ Loaded {len(texts):,} samples")
    
    # Preprocess
    print("\n[2/3] Preprocessing...")
    preprocessor = TextPreprocessor()
    texts_clean = preprocessor.preprocess_batch(texts)
    print(f"  ✓ Preprocessed")
    
    # Fit tokenizer
    print("\n[3/3] Fitting tokenizer...")
    word_tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token='<OOV>')
    word_tokenizer.fit_on_texts(texts_clean)
    
    # Save
    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(exist_ok=True)
    
    # Tokenizer
    tok_path = artifacts_dir / "tokenizer.json"
    with open(tok_path, 'w', encoding='utf-8') as f:
        f.write(word_tokenizer.to_json())
    print(f"  ✓ Saved tokenizer: {tok_path}")
    
    # Config
    config = {
        "max_words": MAX_WORDS,
        "max_len": MAX_LEN,
        "max_char_len": MAX_CHAR_LEN,
        "vocab_size": min(MAX_WORDS, len(word_tokenizer.word_index) + 1)
    }
    cfg_path = artifacts_dir / "config.json"
    with open(cfg_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"  ✓ Saved config: {cfg_path}")
    
    print("\n" + "="*80)
    print("✅ EXPORT COMPLETE")
    print("="*80)


# ============================================================================
# MAIN CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Module 2 DL - Consolidated Scripts',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Combine
    parser_combine = subparsers.add_parser('combine', help='Combine datasets')
    parser_combine.add_argument('--val_split', type=float, default=0.15, help='Validation split ratio')
    
    # Evaluate
    parser_eval = subparsers.add_parser('evaluate', help='Evaluate model')
    parser_eval.add_argument('--model_path', required=True, help='Path to model file')
    parser_eval.add_argument('--test_csv', required=True, help='Path to test CSV')
    parser_eval.add_argument('--output', help='Output JSON file for results')
    
    # Validate
    parser_validate = subparsers.add_parser('validate', help='Validate API')
    parser_validate.add_argument('--api_url', default='http://localhost:8000/predict', help='API URL')
    
    # Find thresholds
    parser_thresh = subparsers.add_parser('find_thresholds', help='Find optimal thresholds')
    parser_thresh.add_argument('--predictions', required=True, help='Path to predictions .npy file')
    parser_thresh.add_argument('--labels', required=True, help='Path to labels CSV')
    parser_thresh.add_argument('--output', help='Output JSON file')
    
    # Export artifacts
    parser_export = subparsers.add_parser('export_artifacts', help='Export tokenizers and config')
    parser_export.add_argument('--source_csv', required=True, help='Source CSV file')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Route to command
    if args.command == 'combine':
        cmd_combine(args)
    elif args.command == 'evaluate':
        cmd_evaluate(args)
    elif args.command == 'validate':
        cmd_validate(args)
    elif args.command == 'find_thresholds':
        cmd_find_thresholds(args)
    elif args.command == 'export_artifacts':
        cmd_export_artifacts(args)


if __name__ == '__main__':
    main()

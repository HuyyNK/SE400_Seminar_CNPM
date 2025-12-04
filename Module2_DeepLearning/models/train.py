"""
Module 2 - Complete Training Script
====================================

Hybrid CNN/BiLSTM Architecture for Toxic Comment Classification

Features:
- Full training pipeline (preprocessing → tokenization → training → evaluation)
- Hybrid architecture: Word-level BiLSTM + Character-level CNN
- Multi-head attention mechanism
- Combined Focal + BCE loss for class imbalance
- Pre-trained GloVe embeddings support
- Comprehensive evaluation metrics

Usage (from Module2_DL root):
    python models/train.py

Training time: 2-3 hours (GPU), 6-8 hours (CPU)
Expected Macro F1: 0.68-0.73
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
module2_root = Path(__file__).parent.parent
sys.path.insert(0, str(module2_root))

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split

# Import consolidated modules
from src.core import ToxicCommentClassifier, focal_loss, combined_focal_bce_loss, load_glove_embeddings
from src.preprocessing import TextPreprocessor, prepare_sequences, prepare_char_sequences, get_default_char_vocab
from src.utils import (
    PROJECT_ROOT, LABEL_COLS, CLASS_WEIGHTS, 
    get_artifact_path, get_model_path,
    GLOVE_EMBEDDING
)

print("=" * 80)
print("🚀 MODULE 2 - TOXIC COMMENT CLASSIFIER")
print("=" * 80)
print("Architecture: Hybrid CNN/BiLSTM with Multi-Head Attention")
print("=" * 80)

# ============================================================================
# PERFORMANCE OPTIMIZATION
# ============================================================================

# Enable XLA JIT compilation for faster execution
try:
    tf.config.optimizer.set_jit(True)
    print("✓ XLA JIT compilation enabled")
except Exception as e:
    print(f"⚠️  XLA JIT not available: {e}")

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    # Data
    'max_words': 50000,
    'max_len': 150,        # OPTION 2: Reduced from 200 (40-50% faster)
    'max_char_len': 200,   # OPTION 2: Reduced from 300 (40-50% faster)
    'test_size': 0.15,
    'random_state': 42,
    
    # Model Architecture (OPTION 2 - BALANCED: optimized for speed)
    'embedding_dim': 300,
    'lstm_units_layer1': 128,  # OPTION 2: Reduced from 256 (50% reduction)
    'lstm_units_layer2': 64,   # OPTION 2: Reduced from 128 (50% reduction)
    'lstm_units_layer3': 32,   # OPTION 2: Reduced from 64 (50% reduction)
    'char_emb_dim': 48,
    'char_num_filters': 128,   # OPTION 2: Reduced from 256 (50% reduction)
    'num_attention_heads': 2,  # OPTION 2: Reduced from 4 (50% reduction)
    'attention_key_dim': 16,   # OPTION 2: Reduced from 32 (50% reduction)
    'trainable_embedding': False,
    'l2_reg': 0.01,
    
    # Training
    'batch_size': 128,
    'epochs': 5,
    'learning_rate': 0.001,
    'patience': 5,
}

print("\n📋 Configuration:")
for key, value in CONFIG.items():
    print(f"   {key}: {value}")

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================

print("\n" + "=" * 80)
print("📊 STEP 1: Loading Data")
print("=" * 80)

train_path = get_artifact_path('train_final.csv')
val_path = get_artifact_path('val_final.csv')

if not train_path.exists() or not val_path.exists():
    print("❌ Data files not found!")
    print(f"   Expected: {train_path}")
    print(f"   Expected: {val_path}")
    print("\n   Please run: python data/prepare_data.py")
    sys.exit(1)

print(f"✓ Loading training data: {train_path}")
df_train = pd.read_csv(train_path)
print(f"✓ Loading validation data: {val_path}")
df_val = pd.read_csv(val_path)

print(f"\n   Training samples: {len(df_train):,}")
print(f"   Validation samples: {len(df_val):,}")

# Extract texts and labels
train_texts = df_train['comment_text'].fillna('').values
train_labels = df_train[LABEL_COLS].values

val_texts = df_val['comment_text'].fillna('').values
val_labels = df_val[LABEL_COLS].values

print(f"\n   Label distribution (training):")
for i, label in enumerate(LABEL_COLS):
    count = train_labels[:, i].sum()
    pct = (count / len(train_labels)) * 100
    print(f"      {label:20s}: {count:6,} ({pct:5.2f}%)")

# ============================================================================
# STEP 2: PREPROCESSING
# ============================================================================

print("\n" + "=" * 80)
print("🧹 STEP 2: Preprocessing Texts")
print("=" * 80)

preprocessor = TextPreprocessor()

print("Processing training texts...")
train_texts_clean = [preprocessor.preprocess(text) for text in train_texts]
print("✓ Training texts preprocessed")

print("Processing validation texts...")
val_texts_clean = [preprocessor.preprocess(text) for text in val_texts]
print("✓ Validation texts preprocessed")

# ============================================================================
# STEP 3: TOKENIZATION - WORD LEVEL
# ============================================================================

print("\n" + "=" * 80)
print("🔤 STEP 3: Word-Level Tokenization")
print("=" * 80)

print(f"Building tokenizer (max_words={CONFIG['max_words']:,})...")
tokenizer = Tokenizer(
    num_words=CONFIG['max_words'],
    oov_token='<UNK>',
    filters='',  # Already cleaned
    lower=False  # Already lowercased
)

tokenizer.fit_on_texts(train_texts_clean)
print(f"✓ Vocabulary size: {len(tokenizer.word_index):,}")

# Prepare sequences
print(f"Converting to sequences (max_len={CONFIG['max_len']})...")
X_train_word = prepare_sequences(train_texts_clean, tokenizer, CONFIG['max_len'])
X_val_word = prepare_sequences(val_texts_clean, tokenizer, CONFIG['max_len'])
print(f"✓ Training sequences shape: {X_train_word.shape}")
print(f"✓ Validation sequences shape: {X_val_word.shape}")

# Save tokenizer
tokenizer_path = get_artifact_path('tokenizer.json')
with open(tokenizer_path, 'w', encoding='utf-8') as f:
    f.write(tokenizer.to_json())
print(f"✓ Tokenizer saved: {tokenizer_path}")

# ============================================================================
# STEP 4: TOKENIZATION - CHARACTER LEVEL
# ============================================================================

print("\n" + "=" * 80)
print("🔡 STEP 4: Character-Level Tokenization")
print("=" * 80)

char_vocab = get_default_char_vocab()
print(f"Character vocabulary size: {len(char_vocab)}")

print(f"Converting to character sequences (max_char_len={CONFIG['max_char_len']})...")
X_train_char = prepare_char_sequences(
    texts=train_texts_clean,
    max_char_len=CONFIG['max_char_len'],
    char_vocab=char_vocab
)
X_val_char = prepare_char_sequences(
    texts=val_texts_clean,
    max_char_len=CONFIG['max_char_len'],
    char_vocab=char_vocab
)
print(f"✓ Training char sequences shape: {X_train_char.shape}")
print(f"✓ Validation char sequences shape: {X_val_char.shape}")

# ============================================================================
# STEP 5: LOAD EMBEDDINGS
# ============================================================================

print("\n" + "=" * 80)
print("📚 STEP 5: Loading Pre-trained Embeddings")
print("=" * 80)

if not GLOVE_EMBEDDING.exists():
    print(f"⚠️  GloVe embeddings not found: {GLOVE_EMBEDDING}")
    print("   Training without pre-trained embeddings...")
    embedding_matrix = None
else:
    print(f"Loading GloVe embeddings: {GLOVE_EMBEDDING}")
    embedding_matrix = load_glove_embeddings(
        word_index=tokenizer.word_index,
        glove_path=str(GLOVE_EMBEDDING),
        embedding_dim=CONFIG['embedding_dim'],
        max_words=CONFIG['max_words']
    )
    
    # Calculate coverage
    if embedding_matrix is not None:
        num_found = np.count_nonzero(np.any(embedding_matrix != 0, axis=1))
        coverage = (num_found / CONFIG['max_words']) * 100
        print(f"✓ Embedding matrix shape: {embedding_matrix.shape}")
        print(f"✓ Words found: {num_found:,} / {CONFIG['max_words']:,} ({coverage:.2f}%)")

# ============================================================================
# STEP 6: BUILD MODEL
# ============================================================================

print("\n" + "=" * 80)
print("🏗️  STEP 6: Building Model Architecture")
print("=" * 80)

print("Architecture: Hybrid CNN/BiLSTM (Optimized)")
print("   - Word Branch: BiLSTM (256→128→64) + Multi-Head Attention")
print("   - Char Branch: Residual CNN (256 filters, 3/4/5 kernels)")
print("   - Fusion: Gated Dynamic Fusion")
print("   - Optimization: Reduced units for 50-60% faster training")

classifier = ToxicCommentClassifier(
    vocab_size=CONFIG['max_words'],
    embedding_dim=CONFIG['embedding_dim'],
    max_len=CONFIG['max_len'],
    char_vocab_size=len(char_vocab),
    max_char_len=CONFIG['max_char_len'],
    lstm_units_layer1=CONFIG['lstm_units_layer1'],
    lstm_units_layer2=CONFIG['lstm_units_layer2'],
    lstm_units_layer3=CONFIG['lstm_units_layer3'],
    char_emb_dim=CONFIG['char_emb_dim'],
    char_num_filters=CONFIG['char_num_filters'],
    num_attention_heads=CONFIG['num_attention_heads'],
    attention_key_dim=CONFIG['attention_key_dim'],
    trainable_embedding=CONFIG['trainable_embedding'],
    l2_reg=CONFIG['l2_reg']
)

model = classifier.build(embedding_matrix=embedding_matrix)

print(f"\n✓ Model built successfully")
print(f"   Total parameters: {model.count_params():,}")

# Show model summary
print("\nModel Summary:")
model.summary(line_length=100)

# ============================================================================
# STEP 7: COMPILE MODEL
# ============================================================================

print("\n" + "=" * 80)
print("⚙️  STEP 7: Compiling Model")
print("=" * 80)

# Use combined loss (Focal + BCE)
loss_fn = combined_focal_bce_loss(
    gamma=2.0,
    alpha=0.25,
    bce_weight=0.4  # 40% BCE, 60% Focal
)

optimizer = tf.keras.optimizers.Adam(learning_rate=CONFIG['learning_rate'])

model.compile(
    optimizer=optimizer,
    loss=loss_fn,
    metrics=[
        'accuracy',
        tf.keras.metrics.AUC(name='auc'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall')
    ]
)

print("✓ Model compiled")
print(f"   Optimizer: Adam (lr={CONFIG['learning_rate']})")
print(f"   Loss: Combined (Focal + Weighted BCE)")
print(f"   Class weights: {CLASS_WEIGHTS}")

# ============================================================================
# STEP 8: SETUP CALLBACKS
# ============================================================================

print("\n" + "=" * 80)
print("📞 STEP 8: Setting up Callbacks")
print("=" * 80)

# Create models directory
models_dir = PROJECT_ROOT / 'artifacts' / 'models'
models_dir.mkdir(parents=True, exist_ok=True)

# Model checkpoint
checkpoint_path = get_model_path('best_model.h5')
checkpoint = ModelCheckpoint(
    str(checkpoint_path),
    monitor='val_loss',
    mode='min',
    save_best_only=True,
    verbose=1
)

# Early stopping
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=CONFIG['patience'],
    restore_best_weights=True,
    verbose=1
)

# Learning rate reduction
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=2,
    min_lr=1e-6,
    verbose=1
)

callbacks = [checkpoint, early_stop, reduce_lr]

print(f"✓ ModelCheckpoint: {checkpoint_path}")
print(f"✓ EarlyStopping: patience={CONFIG['patience']}")
print(f"✓ ReduceLROnPlateau: factor=0.5, patience=2")

# ============================================================================
# STEP 9: TRAIN MODEL
# ============================================================================

print("\n" + "=" * 80)
print("🚀 STEP 9: Training Model")
print("=" * 80)

print(f"Training for {CONFIG['epochs']} epochs (batch_size={CONFIG['batch_size']})")
print(f"Early stopping after {CONFIG['patience']} epochs without improvement")
print("\nOptimizing data pipeline with tf.data...")

# Create optimized tf.data.Dataset pipeline
train_dataset = tf.data.Dataset.from_tensor_slices((
    {'word_input': X_train_word, 'char_input': X_train_char},
    train_labels
))
train_dataset = train_dataset.cache()  # Cache data in memory
train_dataset = train_dataset.shuffle(buffer_size=10000, seed=CONFIG['random_state'])
train_dataset = train_dataset.batch(CONFIG['batch_size'])
train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)  # Prefetch batches

val_dataset = tf.data.Dataset.from_tensor_slices((
    {'word_input': X_val_word, 'char_input': X_val_char},
    val_labels
))
val_dataset = val_dataset.cache()
val_dataset = val_dataset.batch(CONFIG['batch_size'])
val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)

print("✓ Data pipeline optimized with cache() + prefetch()")
print("\nStarting training...\n")

history = model.fit(
    train_dataset,
    epochs=CONFIG['epochs'],
    validation_data=val_dataset,
    callbacks=callbacks,
    verbose=1
)

print("\n✓ Training completed!")

# ============================================================================
# STEP 10: SAVE FINAL MODEL
# ============================================================================

print("\n" + "=" * 80)
print("💾 STEP 10: Saving Model")
print("=" * 80)

final_model_path = get_model_path('final_model.h5')
model.save(str(final_model_path))
print(f"✓ Final model saved: {final_model_path}")

# Save training history
history_path = PROJECT_ROOT / 'artifacts' / 'reports' / 'training_history.json'
history_path.parent.mkdir(parents=True, exist_ok=True)

import json
history_dict = {
    'loss': [float(x) for x in history.history['loss']],
    'val_loss': [float(x) for x in history.history['val_loss']],
    'accuracy': [float(x) for x in history.history['accuracy']],
    'val_accuracy': [float(x) for x in history.history['val_accuracy']],
    'auc': [float(x) for x in history.history['auc']],
    'val_auc': [float(x) for x in history.history['val_auc']],
}

with open(history_path, 'w') as f:
    json.dump(history_dict, f, indent=2)
print(f"✓ Training history saved: {history_path}")

# ============================================================================
# STEP 11: EVALUATE ON VALIDATION SET
# ============================================================================

print("\n" + "=" * 80)
print("📊 STEP 11: Final Evaluation")
print("=" * 80)

# Load best model
best_model = tf.keras.models.load_model(
    str(checkpoint_path),
    custom_objects={'loss_fn': loss_fn}
)

# Predictions
print("Making predictions on validation set...")
val_preds = best_model.predict([X_val_word, X_val_char], batch_size=256, verbose=1)

# Convert to binary (threshold = 0.5)
val_preds_binary = (val_preds > 0.5).astype(int)

# Calculate metrics per label
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

print("\n📈 Per-Label Metrics:")
print(f"{'Label':<20} {'Precision':>10} {'Recall':>10} {'F1':>10} {'AUC':>10}")
print("-" * 62)

f1_scores = []
for i, label in enumerate(LABEL_COLS):
    precision = precision_score(val_labels[:, i], val_preds_binary[:, i], zero_division=0)
    recall = recall_score(val_labels[:, i], val_preds_binary[:, i], zero_division=0)
    f1 = f1_score(val_labels[:, i], val_preds_binary[:, i], zero_division=0)
    try:
        auc = roc_auc_score(val_labels[:, i], val_preds[:, i])
    except:
        auc = 0.0
    
    f1_scores.append(f1)
    print(f"{label:<20} {precision:>10.4f} {recall:>10.4f} {f1:>10.4f} {auc:>10.4f}")

print("-" * 62)
macro_f1 = np.mean(f1_scores)
print(f"{'Macro Average':<20} {'':<10} {'':<10} {macro_f1:>10.4f}")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("✅ TRAINING COMPLETE")
print("=" * 80)

print(f"\n📁 Saved Files:")
print(f"   Model (best):      {checkpoint_path}")
print(f"   Model (final):     {final_model_path}")
print(f"   Tokenizer:         {tokenizer_path}")
print(f"   Training history:  {history_path}")

print(f"\n📊 Final Performance:")
print(f"   Macro F1 Score:    {macro_f1:.4f}")
print(f"   Target: ≥ 0.68 (Good), ≥ 0.70 (Excellent)")

if macro_f1 >= 0.70:
    print("\n🎉 EXCELLENT performance achieved!")
elif macro_f1 >= 0.68:
    print("\n👍 GOOD performance achieved!")
else:
    print("\n⚠️  Performance below target. Consider:")
    print("   - Training longer (more epochs)")
    print("   - Adjusting class weights")
    print("   - Data augmentation")

print("\n🚀 Next Steps:")
print("   1. Test API: python app.py")
print("   2. Evaluate: python data/scripts.py evaluate --model-path artifacts/models/best_model.h5")
print("   3. Find thresholds: python data/scripts.py find_thresholds")

print("\n" + "=" * 80)

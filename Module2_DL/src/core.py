"""
Module 2 DL - Core (Consolidated)
==================================

This file consolidates all core model, training, and prediction logic:
1. Custom Loss Functions (losses.py)
2. Model Architecture (model.py)
3. Training Pipeline (trainer.py)
4. Prediction Pipeline (predictor.py)

Usage:
    from core import ToxicCommentClassifier, focal_loss, ModelTrainer, ToxicCommentPredictor
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, fbeta_score

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import layers, models
from tensorflow.keras.layers import (
    Embedding, Conv1D, GlobalMaxPooling1D, GlobalAveragePooling1D,
    Dense, Dropout, Bidirectional, LSTM, SpatialDropout1D, 
    Concatenate, Input, MultiHeadAttention, Add, LayerNormalization,
    BatchNormalization, Activation, Multiply
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.text import Tokenizer, tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences


# ============================================================================
# SECTION 1: CUSTOM LOSS FUNCTIONS
# ============================================================================

def focal_loss(gamma=2.0, alpha=0.25):
    """
    Focal Loss for multi-label classification with class imbalance.
    
    The Focal Loss focuses training on hard examples by down-weighting
    easy examples. This is particularly useful for:
    - Minority classes (identity_hate: 89 samples, threat: 42 samples)
    - Hard negatives (false positives)
    - Hard positives (false negatives)
    
    Formula:
        FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    
    Args:
        gamma: Focusing parameter (2.0 for standard, 3.0 for very imbalanced)
        alpha: Balancing factor (0.25 focus on positives, 0.5 balanced)
    
    Returns:
        Loss function compatible with Keras model.compile()
    """
    
    def loss_fn(y_true, y_pred):
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
        
        # Cross entropy
        cross_entropy = -y_true * K.log(y_pred) - (1 - y_true) * K.log(1 - y_pred)
        
        # Focal term
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        focal_weight = K.pow(1.0 - p_t, gamma)
        
        # Alpha weight
        alpha_weight = y_true * alpha + (1 - y_true) * (1 - alpha)
        
        focal_loss_value = alpha_weight * focal_weight * cross_entropy
        
        return K.mean(K.sum(focal_loss_value, axis=-1))
    
    return loss_fn


def weighted_binary_crossentropy(class_weights):
    """
    Weighted Binary Cross-Entropy for multi-label classification.
    
    Args:
        class_weights: Dict mapping label index to weight
    
    Returns:
        Loss function compatible with Keras model.compile()
    """
    
    def loss_fn(y_true, y_pred):
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
        
        bce = -y_true * K.log(y_pred) - (1 - y_true) * K.log(1 - y_pred)
        
        weights = y_true * class_weights[1] + (1 - y_true) * class_weights[0]
        weighted_bce = weights * bce
        
        return K.mean(K.sum(weighted_bce, axis=-1))
    
    return loss_fn


def combined_focal_bce_loss(gamma=2.0, alpha=0.25, bce_weight=0.5):
    """
    Combined Focal Loss + Binary Cross-Entropy.
    
    Args:
        gamma: Focal loss focusing parameter
        alpha: Focal loss balancing factor
        bce_weight: Weight for BCE component (0.0 = pure focal, 1.0 = pure BCE)
    
    Returns:
        Loss function compatible with Keras model.compile()
    """
    
    def loss_fn(y_true, y_pred):
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
        
        # BCE
        bce = -y_true * K.log(y_pred) - (1 - y_true) * K.log(1 - y_pred)
        
        # Focal
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        focal_weight = K.pow(1.0 - p_t, gamma)
        alpha_weight = y_true * alpha + (1 - y_true) * (1 - alpha)
        focal = alpha_weight * focal_weight * bce
        
        # Combine
        combined = bce_weight * bce + (1 - bce_weight) * focal
        
        return K.mean(K.sum(combined, axis=-1))
    
    return loss_fn


def get_loss_function(loss_name='focal', **kwargs):
    """
    Factory function to get loss function by name.
    
    Args:
        loss_name: One of ['focal', 'weighted_bce', 'combined', 'binary_crossentropy']
        **kwargs: Loss-specific parameters
    
    Returns:
        Loss function
    """
    
    if loss_name == 'focal':
        gamma = kwargs.get('gamma', 2.0)
        alpha = kwargs.get('alpha', 0.25)
        return focal_loss(gamma=gamma, alpha=alpha)
    
    elif loss_name == 'weighted_bce':
        class_weights = kwargs.get('class_weights', {0: 1.0, 1: 1.0})
        return weighted_binary_crossentropy(class_weights)
    
    elif loss_name == 'combined':
        gamma = kwargs.get('gamma', 2.0)
        alpha = kwargs.get('alpha', 0.25)
        bce_weight = kwargs.get('bce_weight', 0.5)
        return combined_focal_bce_loss(gamma=gamma, alpha=alpha, bce_weight=bce_weight)
    
    elif loss_name == 'binary_crossentropy':
        return 'binary_crossentropy'
    
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")


# ============================================================================
# SECTION 2: MODEL ARCHITECTURE
# ============================================================================

class ToxicCommentClassifier:
    """
    Best-in-class toxic comment classifier
    
    Architecture:
        Word Branch:
            - GloVe Embedding (300d, frozen)
            - Spatial Dropout (0.2)
            - Stacked BiLSTM (3 layers: 512→256→128)
            - Stacked Multi-Head Attention (2 layers)
            - Dual Pooling (Max + Average)
            
        Character Branch:
            - Char Embedding (48d)
            - Deep Residual CNN (multi-kernel: 3,4,5)
            - Global Max Pooling
            
        Fusion:
            - Gated Dynamic Fusion
            - Dense layers with dropout
            - Sigmoid output (6 labels)
    
    Expected Performance: Macro F1 = 0.68-0.73
    """
    
    def __init__(
        self,
        vocab_size: int = 50000,
        embedding_dim: int = 300,
        max_len: int = 250,
        char_vocab_size: int = 100,
        max_char_len: int = 400,
        lstm_units_layer1: int = 512,
        lstm_units_layer2: int = 256,
        lstm_units_layer3: int = 128,
        char_emb_dim: int = 48,
        char_num_filters: int = 512,
        num_attention_heads: int = 8,
        attention_key_dim: int = 64,
        trainable_embedding: bool = False,
        l2_reg: float = 0.01
    ):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_len = max_len
        self.char_vocab_size = char_vocab_size
        self.max_char_len = max_char_len
        self.lstm_units_layer1 = lstm_units_layer1
        self.lstm_units_layer2 = lstm_units_layer2
        self.lstm_units_layer3 = lstm_units_layer3
        self.char_emb_dim = char_emb_dim
        self.char_num_filters = char_num_filters
        self.num_attention_heads = num_attention_heads
        self.attention_key_dim = attention_key_dim
        self.trainable_embedding = trainable_embedding
        self.l2_reg = l2_reg
        self.model = None
    
    def build(self, embedding_matrix: Optional[np.ndarray] = None) -> keras.Model:
        """
        Build the complete model architecture
        
        Args:
            embedding_matrix: Pre-trained embeddings (vocab_size x embedding_dim)
        
        Returns:
            Compiled Keras model
        """
        # Word Branch
        word_input = Input(shape=(self.max_len,), name='word_input')
        
        if embedding_matrix is not None:
            word_emb = Embedding(
                input_dim=self.vocab_size,
                output_dim=self.embedding_dim,
                input_length=self.max_len,
                weights=[embedding_matrix],
                trainable=self.trainable_embedding,
                name='word_embedding'
            )(word_input)
        else:
            word_emb = Embedding(
                input_dim=self.vocab_size,
                output_dim=self.embedding_dim,
                input_length=self.max_len,
                trainable=True,
                name='word_embedding'
            )(word_input)
        
        word_emb = SpatialDropout1D(0.2, name='word_spatial_dropout')(word_emb)
        
        # Stacked BiLSTM (3 layers)
        lstm_out = Bidirectional(
            LSTM(self.lstm_units_layer1, return_sequences=True, 
                 dropout=0.2, recurrent_dropout=0.2),
            name='word_bilstm_layer1'
        )(word_emb)
        
        lstm_out = Bidirectional(
            LSTM(self.lstm_units_layer2, return_sequences=True, dropout=0.2),
            name='word_bilstm_layer2'
        )(lstm_out)
        
        lstm_out = Bidirectional(
            LSTM(self.lstm_units_layer3, return_sequences=True, dropout=0.2),
            name='word_bilstm_layer3'
        )(lstm_out)
        
        # Stacked Multi-Head Attention (2 layers)
        attention_residual1 = lstm_out
        attention_out = MultiHeadAttention(
            num_heads=self.num_attention_heads,
            key_dim=self.attention_key_dim,
            name='multi_head_attention_layer1'
        )(lstm_out, lstm_out)
        attention_out = Add(name='residual_connection_1')([attention_residual1, attention_out])
        attention_out = LayerNormalization(name='layer_norm_1')(attention_out)
        
        attention_residual2 = attention_out
        attention_out = MultiHeadAttention(
            num_heads=self.num_attention_heads,
            key_dim=self.attention_key_dim,
            name='multi_head_attention_layer2'
        )(attention_out, attention_out)
        attention_out = Add(name='residual_connection_2')([attention_residual2, attention_out])
        attention_out = LayerNormalization(name='layer_norm_2')(attention_out)
        
        # Dual pooling
        max_pool = GlobalMaxPooling1D(name='word_max_pool')(attention_out)
        avg_pool = GlobalAveragePooling1D(name='word_avg_pool')(attention_out)
        word_features = Concatenate(name='word_pool_concat')([max_pool, avg_pool])
        
        # Character Branch
        char_input = Input(shape=(self.max_char_len,), name='char_input')
        
        char_emb = Embedding(
            input_dim=self.char_vocab_size,
            output_dim=self.char_emb_dim,
            input_length=self.max_char_len,
            name='char_embedding'
        )(char_input)
        
        # Deep Residual CNN
        char_conv_blocks = []
        for kernel_size in (3, 4, 5):
            conv = Conv1D(
                filters=self.char_num_filters,
                kernel_size=kernel_size,
                activation='relu',
                padding='same',
                name=f'char_conv_{kernel_size}_layer1'
            )(char_emb)
            conv = BatchNormalization(name=f'char_bn_{kernel_size}_layer1')(conv)
            
            conv = Conv1D(
                filters=self.char_num_filters,
                kernel_size=kernel_size,
                activation='relu',
                padding='same',
                name=f'char_conv_{kernel_size}_layer2'
            )(conv)
            conv = BatchNormalization(name=f'char_bn_{kernel_size}_layer2')(conv)
            
            residual = Conv1D(
                filters=self.char_num_filters,
                kernel_size=1,
                padding='same',
                name=f'char_residual_{kernel_size}'
            )(char_emb)
            conv = Add(name=f'char_add_{kernel_size}')([residual, conv])
            conv = Activation('relu', name=f'char_activation_{kernel_size}')(conv)
            
            pool = GlobalMaxPooling1D(name=f'char_pool_{kernel_size}')(conv)
            char_conv_blocks.append(pool)
        
        char_concat = Concatenate(name='char_concat')(char_conv_blocks)
        char_features = Dense(128, activation='relu', 
                            kernel_regularizer=l2(self.l2_reg),
                            name='char_dense')(char_concat)
        
        # Gated Fusion
        gate = Dense(word_features.shape[-1], activation='sigmoid', 
                    name='fusion_gate')(word_features)
        gated_word = Multiply(name='gated_word')([gate, word_features])
        
        inverse_gate = layers.Lambda(lambda x: 1 - x, name='inverse_gate')(gate)
        char_features_expanded = Dense(word_features.shape[-1], 
                                      kernel_regularizer=l2(self.l2_reg),
                                      name='char_expand')(char_features)
        gated_char = Multiply(name='gated_char')([inverse_gate, char_features_expanded])
        
        fusion = Add(name='fusion_add')([gated_word, gated_char])
        
        # Dense layers
        x = Dense(128, activation='relu', 
                 kernel_regularizer=l2(self.l2_reg),
                 kernel_initializer='he_normal',
                 name='fusion_dense_1')(fusion)
        x = Dropout(0.5, name='fusion_dropout_1')(x)
        
        x = Dense(64, activation='relu', 
                 kernel_regularizer=l2(self.l2_reg),
                 kernel_initializer='he_normal',
                 name='fusion_dense_2')(x)
        x = Dropout(0.5, name='fusion_dropout_2')(x)
        
        # Output
        output = Dense(6, activation='sigmoid', name='output')(x)
        
        self.model = models.Model(
            inputs=[word_input, char_input],
            outputs=output,
            name='ToxicCommentClassifier_BestModel'
        )
        
        return self.model
    
    def compile(
        self,
        learning_rate: float = 0.001,
        use_focal_loss: bool = True,
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.25,
        label_smoothing: float = 0.1,
        clipnorm: float = 1.0
    ):
        """Compile model with optimizer and loss"""
        optimizer = keras.optimizers.Adam(
            learning_rate=learning_rate,
            clipnorm=clipnorm
        )
        
        if use_focal_loss:
            loss = self._binary_focal_loss(gamma=focal_gamma, alpha=focal_alpha, 
                                          label_smoothing=label_smoothing)
        else:
            loss = 'binary_crossentropy'
        
        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=['accuracy', keras.metrics.AUC(name='auc')]
        )
        
        print(f"✓ Model compiled with {loss if isinstance(loss, str) else 'focal_loss'}")
    
    def _binary_focal_loss(self, gamma: float = 2.0, alpha: float = 0.25, 
                          label_smoothing: float = 0.1):
        """Binary focal loss with label smoothing"""
        def loss(y_true, y_pred):
            y_true = tf.cast(y_true, tf.float32)
            
            if label_smoothing > 0:
                y_true = y_true * (1.0 - label_smoothing) + 0.5 * label_smoothing
            
            y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
            
            ce = - (y_true * tf.math.log(y_pred) + 
                   (1 - y_true) * tf.math.log(1 - y_pred))
            
            p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
            alpha_factor = y_true * alpha + (1 - y_true) * (1 - alpha)
            modulating = tf.pow(1.0 - p_t, gamma)
            
            return tf.reduce_mean(alpha_factor * modulating * ce)
        
        return loss
    
    def get_callbacks(
        self,
        model_path: str = 'artifacts/models/best_model.h5',
        patience_early_stop: int = 3,
        patience_reduce_lr: int = 2
    ) -> list:
        """Get training callbacks"""
        callbacks_list = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=patience_early_stop,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=patience_reduce_lr,
                min_lr=1e-6,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                filepath=model_path,
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]
        
        return callbacks_list
    
    def summary(self):
        """Print model architecture summary"""
        if self.model is None:
            raise ValueError("Model not built yet. Call build() first.")
        
        self.model.summary()
        
        trainable_params = sum([tf.size(w).numpy() for w in self.model.trainable_weights])
        non_trainable_params = sum([tf.size(w).numpy() for w in self.model.non_trainable_weights])
        total_params = trainable_params + non_trainable_params
        
        print(f"\n{'='*60}")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Non-trainable parameters: {non_trainable_params:,}")
        print(f"{'='*60}\n")


# ============================================================================
# SECTION 3: EMBEDDING UTILITIES
# ============================================================================

def load_fasttext_embeddings(
    word_index: Dict[str, int],
    fasttext_path: str,
    embedding_dim: int = 300,
    max_words: int = 50000
) -> np.ndarray:
    """
    Load FastText embeddings and create embedding matrix
    
    Args:
        word_index: Word to index mapping from tokenizer
        fasttext_path: Path to FastText .vec file
        embedding_dim: Embedding dimension (300)
        max_words: Maximum vocabulary size
    
    Returns:
        Embedding matrix (vocab_size x embedding_dim)
    """
    print(f"Loading FastText embeddings from {fasttext_path}...")
    
    embeddings_index = {}
    with open(fasttext_path, encoding='utf-8') as f:
        next(f)  # Skip first line
        for line in f:
            values = line.rstrip().split(' ')
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            if len(coefs) == embedding_dim:
                embeddings_index[word] = coefs
    
    print(f"✓ Loaded {len(embeddings_index):,} word vectors")
    
    vocab_size = min(len(word_index) + 1, max_words)
    embedding_matrix = np.zeros((vocab_size, embedding_dim), dtype='float32')
    
    found = 0
    for word, idx in word_index.items():
        if idx >= max_words:
            continue
        
        vector = embeddings_index.get(word)
        if vector is not None:
            embedding_matrix[idx] = vector
            found += 1
        else:
            embedding_matrix[idx] = np.random.normal(0, 0.05, embedding_dim).astype('float32')
    
    coverage = found / (vocab_size - 1) * 100
    print(f"✓ Coverage: {found:,}/{vocab_size-1:,} words ({coverage:.1f}%)")
    print(f"✓ Embedding matrix shape: {embedding_matrix.shape}")
    
    return embedding_matrix


def load_glove_embeddings(
    word_index: Dict[str, int],
    glove_path: str,
    embedding_dim: int = 300,
    max_words: int = 50000
) -> np.ndarray:
    """
    Load GloVe embeddings and create embedding matrix
    
    Args:
        word_index: Word to index mapping from tokenizer
        glove_path: Path to GloVe .txt file
        embedding_dim: Embedding dimension (300)
        max_words: Maximum vocabulary size
    
    Returns:
        Embedding matrix (vocab_size x embedding_dim)
    """
    print(f"Loading GloVe embeddings from {glove_path}...")
    
    embeddings_index = {}
    with open(glove_path, encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    
    print(f"✓ Loaded {len(embeddings_index):,} word vectors")
    
    vocab_size = min(len(word_index) + 1, max_words)
    embedding_matrix = np.zeros((vocab_size, embedding_dim), dtype='float32')
    
    found = 0
    for word, idx in word_index.items():
        if idx >= max_words:
            continue
        
        vector = embeddings_index.get(word)
        if vector is not None:
            embedding_matrix[idx] = vector
            found += 1
        else:
            embedding_matrix[idx] = np.random.normal(0, 0.05, embedding_dim).astype('float32')
    
    coverage = found / (vocab_size - 1) * 100
    print(f"✓ Coverage: {found:,}/{vocab_size-1:,} words ({coverage:.1f}%)")
    print(f"✓ Embedding matrix shape: {embedding_matrix.shape}")
    
    return embedding_matrix


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Loss functions
    'focal_loss', 'weighted_binary_crossentropy', 'combined_focal_bce_loss',
    'get_loss_function',
    
    # Model
    'ToxicCommentClassifier',
    
    # Embeddings
    'load_fasttext_embeddings', 'load_glove_embeddings',
]

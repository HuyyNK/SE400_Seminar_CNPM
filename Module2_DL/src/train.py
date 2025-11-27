"""
Module 2: Script huấn luyện mô hình CNN hoặc BiLSTM
Hỗ trợ:
- Load và tiền xử lý dữ liệu
- Tạo tokenizer và embedding matrix
- Train với callbacks
- Lưu model và artifacts
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.utils.class_weight import compute_class_weight

# Import từ các module khác
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from preprocess import TextPreprocessor, load_and_split_data, prepare_sequences
from models import (
    create_embedding_matrix, 
    build_cnn_model, 
    build_bilstm_model,
    get_callbacks
)


def plot_learning_curves(history, output_dir: str, model_type: str):
    """
    Vẽ learning curves (loss và metrics) để hiểu khi nào model hội tụ
    
    Args:
        history: History object từ model.fit()
        output_dir: Thư mục lưu plot
        model_type: Loại model (cnn/bilstm)
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'{model_type.upper()} Learning Curves', fontsize=16, fontweight='bold')
    
    # Plot 1: Loss
    axes[0, 0].plot(history.history['loss'], label='Train Loss', linewidth=2)
    axes[0, 0].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Binary Crossentropy Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: AUC
    axes[0, 1].plot(history.history['auc'], label='Train AUC', linewidth=2)
    axes[0, 1].plot(history.history['val_auc'], label='Val AUC', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('AUC')
    axes[0, 1].set_title('Area Under ROC Curve')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Precision
    axes[1, 0].plot(history.history['precision'], label='Train Precision', linewidth=2)
    axes[1, 0].plot(history.history['val_precision'], label='Val Precision', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].set_title('Precision Score')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Recall
    axes[1, 1].plot(history.history['recall'], label='Train Recall', linewidth=2)
    axes[1, 1].plot(history.history['val_recall'], label='Val Recall', linewidth=2)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Recall')
    axes[1, 1].set_title('Recall Score')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Lưu figure
    curves_path = os.path.join(output_dir, f'learning_curves_{model_type}.png')
    plt.savefig(curves_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved learning curves to {curves_path}")
    plt.close()


def compute_class_weights(y_train: np.ndarray) -> dict:
    """
    Tính class weights cho từng nhãn để xử lý imbalance
    
    Args:
        y_train: Labels array shape (n_samples, 6)
    
    Returns:
        Dict của class weights cho từng label
    """
    class_weights_dict = {}
    
    for i in range(y_train.shape[1]):
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.array([0, 1]),
            y=y_train[:, i]
        )
        class_weights_dict[i] = {0: class_weights[0], 1: class_weights[1]}
    
    return class_weights_dict


def train_model(
    model_type: str = 'cnn',
    train_csv: str = '../Data/train.csv',
    slang_csv: str = '../Data/slang.csv',
    glove_path: str = '../embeddings/glove.6B.300d.txt',
    max_words: int = 50000,
    max_len: int = 250,
    embedding_dim: int = 300,
    batch_size: int = 256,
    epochs: int = 20,
    output_dir: str = '../artifacts'
):
    """
    Pipeline huấn luyện đầy đủ
    
    Args:
        model_type: 'cnn' hoặc 'bilstm'
        train_csv: Đường dẫn tới train.csv
        slang_csv: Đường dẫn tới slang.csv
        glove_path: Đường dẫn tới GloVe embeddings
        max_words: Số từ tối đa trong vocabulary
        max_len: Độ dài sequence
        embedding_dim: Số chiều embedding
        batch_size: Batch size cho training
        epochs: Số epochs tối đa
        output_dir: Thư mục lưu artifacts
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print(f"Training {model_type.upper()} Model")
    print("="*60)
    
    # 1. Load và chia dữ liệu
    print("\n[1/7] Loading and splitting data...")
    train_df, val_df, test_df = load_and_split_data(
        train_csv_path=train_csv,
        test_size=0.2,
        val_size=0.1
    )
    
    # 2. Tiền xử lý văn bản
    print("\n[2/7] Preprocessing text...")
    preprocessor = TextPreprocessor(slang_dict_path=slang_csv, remove_stopwords=False)
    
    train_texts = preprocessor.preprocess_batch(train_df['comment_text'].tolist())
    val_texts = preprocessor.preprocess_batch(val_df['comment_text'].tolist())
    test_texts = preprocessor.preprocess_batch(test_df['comment_text'].tolist())
    
    # Labels
    label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    y_train = train_df[label_cols].values
    y_val = val_df[label_cols].values
    y_test = test_df[label_cols].values
    
    print(f"✓ Train samples: {len(train_texts)}")
    print(f"✓ Val samples: {len(val_texts)}")
    print(f"✓ Test samples: {len(test_texts)}")
    
    # 3. Tokenization
    print("\n[3/7] Tokenizing and creating sequences...")
    tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
    tokenizer.fit_on_texts(train_texts)
    
    X_train = prepare_sequences(train_texts, tokenizer, max_len)
    X_val = prepare_sequences(val_texts, tokenizer, max_len)
    X_test = prepare_sequences(test_texts, tokenizer, max_len)
    
    vocab_size = min(len(tokenizer.word_index) + 1, max_words)
    print(f"✓ Vocabulary size: {vocab_size}")
    print(f"✓ Sequence shape: {X_train.shape}")
    
    # 4. Tạo embedding matrix từ GloVe
    print("\n[4/7] Creating embedding matrix...")
    if os.path.exists(glove_path):
        embedding_matrix = create_embedding_matrix(
            word_index=tokenizer.word_index,
            embedding_path=glove_path,
            embedding_dim=embedding_dim,
            max_words=max_words
        )
    else:
        print(f"⚠ GloVe file not found at {glove_path}")
        print("  Training với random embedding initialization...")
        embedding_matrix = None
    
    # 5. Build model
    print("\n[5/7] Building model...")
    if model_type.lower() == 'cnn':
        model = build_cnn_model(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            max_len=max_len,
            embedding_matrix=embedding_matrix,
            num_filters=256,
            kernel_sizes=(3, 4, 5),
            trainable_embedding=False
        )
    elif model_type.lower() == 'bilstm':
        model = build_bilstm_model(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            max_len=max_len,
            embedding_matrix=embedding_matrix,
            lstm_units=128,
            trainable_embedding=False,
            use_two_layers=False
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    # Compile
    model.compile(
        loss='binary_crossentropy',
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        metrics=[
            keras.metrics.AUC(name='auc', multi_label=True),
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall')
        ]
    )
    
    model.summary()
    
    # 6. Train
    print("\n[6/7] Training model...")
    model_path = os.path.join(output_dir, f'toxic_{model_type}_model.h5')
    
    callback_list = get_callbacks(
        model_path=model_path,
        patience_early=5,
        patience_lr=2,
        use_early_stopping=True  # Bật EarlyStopping để tự động dừng khi model hội tụ
    )
    
    # Tính class weights (tùy chọn)
    # class_weights = compute_class_weights(y_train)
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callback_list,
        verbose=1
        # class_weight=class_weights  # Uncomment nếu cần
    )
    
    # 7. Lưu artifacts
    print("\n[7/7] Saving artifacts...")
    
    # Lưu tokenizer
    tokenizer_config = tokenizer.to_json()
    tokenizer_path = os.path.join(output_dir, f'tokenizer_{model_type}.json')
    with open(tokenizer_path, 'w', encoding='utf-8') as f:
        f.write(tokenizer_config)
    print(f"✓ Saved tokenizer to {tokenizer_path}")
    
    # Lưu config
    config = {
        'model_type': model_type,
        'max_len': max_len,
        'max_words': max_words,
        'embedding_dim': embedding_dim,
        'vocab_size': vocab_size,
        'label_cols': label_cols,
        'batch_size': batch_size,
        'epochs': epochs,
        'best_epoch': len(history.history['loss']) - 5  # Epoch tốt nhất (EarlyStopping patience=5)
    }
    config_path = os.path.join(output_dir, f'config_{model_type}.json')
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)
    print(f"✓ Saved config to {config_path}")
    
    # Lưu history
    history_path = os.path.join(output_dir, f'history_{model_type}.json')
    history_dict = history.history.copy()
    history_dict['best_epoch'] = len(history.history['loss']) - 5
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump(history_dict, f, indent=2)
    print(f"✓ Saved training history to {history_path}")
    
    # Vẽ learning curves
    print("\n[8/8] Plotting learning curves...")
    plot_learning_curves(history, output_dir, model_type)
    
    print("\n" + "="*60)
    print("Training Complete!")
    print(f"Best model saved to: {model_path}")
    print(f"Training stopped at epoch: {len(history.history['loss'])}")
    print(f"Best epoch (val_loss): {len(history.history['loss']) - 5}")
    print("="*60)
    
    return model, tokenizer, config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train CNN or BiLSTM model')
    parser.add_argument('--model', type=str, default='cnn', choices=['cnn', 'bilstm'],
                        help='Model type: cnn or bilstm')
    parser.add_argument('--train_csv', type=str, default='../Data/train.csv',
                        help='Path to train.csv')
    parser.add_argument('--slang_csv', type=str, default='../Data/slang.csv',
                        help='Path to slang.csv')
    parser.add_argument('--glove', type=str, default='../embeddings/glove.6B.300d.txt',
                        help='Path to GloVe embeddings')
    parser.add_argument('--max_words', type=int, default=50000,
                        help='Max vocabulary size')
    parser.add_argument('--max_len', type=int, default=250,
                        help='Max sequence length')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Max epochs')
    parser.add_argument('--output', type=str, default='../artifacts',
                        help='Output directory for artifacts')
    
    args = parser.parse_args()
    
    train_model(
        model_type=args.model,
        train_csv=args.train_csv,
        slang_csv=args.slang_csv,
        glove_path=args.glove,
        max_words=args.max_words,
        max_len=args.max_len,
        batch_size=args.batch_size,
        epochs=args.epochs,
        output_dir=args.output
    )

"""
Module 2: Định nghĩa kiến trúc mô hình CNN và BiLSTM
- CNN: Nhanh, hiệu quả, tốt cho việc bắt n-gram độc hại
- BiLSTM: Hiểu ngữ cảnh tuần tự tốt hơn
"""

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.layers import (
    Embedding, Conv1D, GlobalMaxPooling1D, 
    Dense, Dropout, Bidirectional, LSTM, 
    SpatialDropout1D, Concatenate, Input
)
from typing import Tuple, Dict, Optional


def create_embedding_matrix(
    word_index: Dict[str, int],
    embedding_path: str,
    embedding_dim: int = 300,
    max_words: int = 50000
) -> np.ndarray:
    """
    Tạo embedding matrix từ GloVe pre-trained embeddings
    
    Args:
        word_index: Dict từ Keras Tokenizer (word -> index)
        embedding_path: Đường dẫn tới file GloVe (ví dụ: glove.6B.300d.txt)
        embedding_dim: Số chiều embedding (300 cho GloVe 300d)
        max_words: Số từ tối đa trong vocabulary
    
    Returns:
        embedding_matrix: numpy array shape (vocab_size, embedding_dim)
    """
    print(f"Loading GloVe embeddings from {embedding_path}...")
    
    # Load GloVe
    embeddings_index = {}
    with open(embedding_path, encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    
    print(f"✓ Found {len(embeddings_index)} word vectors in GloVe")
    
    # Tạo embedding matrix
    vocab_size = min(len(word_index) + 1, max_words)
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    
    found = 0
    for word, i in word_index.items():
        if i >= max_words:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            found += 1
        else:
            # OOV: khởi tạo ngẫu nhiên nhỏ
            embedding_matrix[i] = np.random.normal(0, 0.05, embedding_dim)
    
    print(f"✓ Filled {found}/{vocab_size-1} words from GloVe ({found/(vocab_size-1)*100:.1f}%)")
    print(f"✓ Embedding matrix shape: {embedding_matrix.shape}")
    
    return embedding_matrix


def build_cnn_model(
    vocab_size: int,
    embedding_dim: int,
    max_len: int,
    embedding_matrix: Optional[np.ndarray] = None,
    num_filters: int = 256,
    kernel_sizes: Tuple[int, ...] = (3, 4, 5),
    trainable_embedding: bool = False
) -> keras.Model:
    """
    Xây dựng mô hình CNN cho phân loại đa nhãn
    
    Args:
        vocab_size: Kích thước vocabulary
        embedding_dim: Số chiều embedding (300 cho GloVe)
        max_len: Độ dài tối đa của sequence
        embedding_matrix: Ma trận embedding từ GloVe (nếu có)
        num_filters: Số filters cho mỗi kernel size
        kernel_sizes: Tuple các kích thước kernel (ví dụ: (3,4,5) cho trigram, 4-gram, 5-gram)
        trainable_embedding: Có train embedding layer không (mặc định False để giữ GloVe)
    
    Returns:
        Keras Model
    """
    # Input
    input_text = Input(shape=(max_len,), name='input_text')
    
    # Embedding layer
    if embedding_matrix is not None:
        embedding = Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            input_length=max_len,
            weights=[embedding_matrix],
            trainable=trainable_embedding,
            name='embedding'
        )(input_text)
    else:
        embedding = Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            input_length=max_len,
            trainable=True,
            name='embedding'
        )(input_text)
    
    # Spatial Dropout (giảm overfitting)
    x = SpatialDropout1D(0.2)(embedding)
    
    # Multi-kernel CNN
    conv_blocks = []
    for kernel_size in kernel_sizes:
        conv = Conv1D(
            filters=num_filters,
            kernel_size=kernel_size,
            activation='relu',
            padding='same',
            name=f'conv_{kernel_size}'
        )(x)
        conv = GlobalMaxPooling1D(name=f'pool_{kernel_size}')(conv)
        conv_blocks.append(conv)
    
    # Concatenate các conv blocks
    if len(conv_blocks) > 1:
        x = Concatenate(name='concat')(conv_blocks)
    else:
        x = conv_blocks[0]
    
    # Dense layers
    x = Dense(128, activation='relu', name='dense_1')(x)
    x = Dropout(0.5, name='dropout_1')(x)
    x = Dense(64, activation='relu', name='dense_2')(x)
    x = Dropout(0.5, name='dropout_2')(x)
    
    # Output layer (6 nhãn, sigmoid cho multi-label)
    output = Dense(6, activation='sigmoid', name='output')(x)
    
    # Build model
    model = models.Model(inputs=input_text, outputs=output, name='CNN_Toxic_Classifier')
    
    return model


def build_bilstm_model(
    vocab_size: int,
    embedding_dim: int,
    max_len: int,
    embedding_matrix: Optional[np.ndarray] = None,
    lstm_units: int = 128,
    trainable_embedding: bool = False,
    use_two_layers: bool = False
) -> keras.Model:
    """
    Xây dựng mô hình Bidirectional LSTM cho phân loại đa nhãn
    
    Args:
        vocab_size: Kích thước vocabulary
        embedding_dim: Số chiều embedding (300 cho GloVe)
        max_len: Độ dài tối đa của sequence
        embedding_matrix: Ma trận embedding từ GloVe (nếu có)
        lstm_units: Số units trong LSTM layer
        trainable_embedding: Có train embedding layer không
        use_two_layers: Dùng 2 tầng BiLSTM (chậm hơn nhưng mạnh hơn)
    
    Returns:
        Keras Model
    """
    # Input
    input_text = Input(shape=(max_len,), name='input_text')
    
    # Embedding layer
    if embedding_matrix is not None:
        embedding = Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            input_length=max_len,
            weights=[embedding_matrix],
            trainable=trainable_embedding,
            name='embedding'
        )(input_text)
    else:
        embedding = Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            input_length=max_len,
            trainable=True,
            name='embedding'
        )(input_text)
    
    # Spatial Dropout
    x = SpatialDropout1D(0.2)(embedding)
    
    # BiLSTM layers
    if use_two_layers:
        x = Bidirectional(LSTM(lstm_units, return_sequences=True), name='bilstm_1')(x)
        x = Dropout(0.3)(x)
        x = Bidirectional(LSTM(lstm_units // 2, return_sequences=False), name='bilstm_2')(x)
    else:
        x = Bidirectional(LSTM(lstm_units, return_sequences=False), name='bilstm')(x)
    
    x = Dropout(0.3)(x)
    
    # Dense layers
    x = Dense(128, activation='relu', name='dense_1')(x)
    x = Dropout(0.5, name='dropout_1')(x)
    x = Dense(64, activation='relu', name='dense_2')(x)
    x = Dropout(0.5, name='dropout_2')(x)
    
    # Output layer
    output = Dense(6, activation='sigmoid', name='output')(x)
    
    # Build model
    model = models.Model(inputs=input_text, outputs=output, name='BiLSTM_Toxic_Classifier')
    
    return model


def get_callbacks(
    model_path: str,
    patience_early: int = 3,
    patience_lr: int = 2,
    factor_lr: float = 0.5,
    min_lr: float = 1e-6,
    use_early_stopping: bool = True
) -> list:
    """
    Tạo danh sách callbacks cho training
    
    Args:
        model_path: Đường dẫn lưu model tốt nhất
        patience_early: Patience cho EarlyStopping
        patience_lr: Patience cho ReduceLROnPlateau
        factor_lr: Factor giảm learning rate
        min_lr: Learning rate tối thiểu
        use_early_stopping: Có dùng EarlyStopping không
    
    Returns:
        List of callbacks
    """
    callback_list = []
    
    if use_early_stopping:
        callback_list.append(
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=patience_early,
                restore_best_weights=True,
                verbose=1
            )
        )
    
    callback_list.extend([
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=factor_lr,
            patience=patience_lr,
            min_lr=min_lr,
            verbose=1
        ),
        callbacks.ModelCheckpoint(
            filepath=model_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ])
    
    return callback_list


if __name__ == "__main__":
    # Test model architectures
    print("\n=== Testing CNN Model ===")
    cnn_model = build_cnn_model(
        vocab_size=50000,
        embedding_dim=300,
        max_len=250,
        num_filters=128,
        kernel_sizes=(3, 4, 5)
    )
    cnn_model.summary()
    
    print("\n=== Testing BiLSTM Model ===")
    bilstm_model = build_bilstm_model(
        vocab_size=50000,
        embedding_dim=300,
        max_len=250,
        lstm_units=128,
        use_two_layers=False
    )
    bilstm_model.summary()

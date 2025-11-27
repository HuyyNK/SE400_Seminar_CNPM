"""
Module 2: Tiền xử lý dữ liệu cho Deep Learning
- Làm sạch văn bản (lower-case, bỏ URL/mention/emoji)
- Chuẩn hóa slang dựa trên slang.csv
- Tokenization và padding cho Keras
"""

import re
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import nltk
from nltk.corpus import stopwords

# Tải stopwords nếu chưa có
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)


class TextPreprocessor:
    """
    Bộ tiền xử lý văn bản cho mô hình Deep Learning
    """
    
    def __init__(self, slang_dict_path: str = None, remove_stopwords: bool = False):
        """
        Args:
            slang_dict_path: Đường dẫn tới slang.csv
            remove_stopwords: Có loại bỏ stopwords không (mặc định False vì DL có thể học được)
        """
        self.slang_dict = {}
        self.remove_stopwords = remove_stopwords
        self.stop_words = set(stopwords.words('english')) if remove_stopwords else set()
        
        if slang_dict_path:
            self.load_slang_dict(slang_dict_path)
    
    def load_slang_dict(self, path: str):
        """Load từ điển slang từ CSV"""
        try:
            df = pd.read_csv(path, encoding='utf-8')
            # Giả định có cột 'slang' và 'normalized' hoặc tương tự
            if 'slang' in df.columns:
                for _, row in df.iterrows():
                    slang = str(row.get('slang', '')).lower().strip()
                    # Nếu có cột normalized, dùng; không thì để trống
                    normalized = str(row.get('normalized', slang)).lower().strip()
                    if slang:
                        self.slang_dict[slang] = normalized
            print(f"✓ Loaded {len(self.slang_dict)} slang terms")
        except Exception as e:
            print(f"⚠ Could not load slang dict: {e}")
    
    def clean_text(self, text: str) -> str:
        """
        Làm sạch văn bản cơ bản
        - Lower-case
        - Bỏ URLs
        - Bỏ mentions (@user)
        - Bỏ emoji/special chars (giữ dấu câu cơ bản)
        - Chuẩn hóa khoảng trắng
        """
        if not isinstance(text, str):
            return ""
        
        # Lower-case
        text = text.lower()
        
        # Bỏ URLs
        text = re.sub(r'http\S+|www\.\S+', '', text)
        
        # Bỏ mentions
        text = re.sub(r'@\w+', '', text)
        
        # Bỏ emoji và ký tự đặc biệt (giữ chữ, số, dấu câu cơ bản)
        text = re.sub(r'[^\w\s\.\,\!\?\-\']', ' ', text)
        
        # Chuẩn hóa khoảng trắng
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def normalize_slang(self, text: str) -> str:
        """
        Thay thế slang bằng từ chuẩn hóa
        """
        if not self.slang_dict:
            return text
        
        words = text.split()
        normalized_words = []
        
        for word in words:
            # Kiểm tra slang dict
            normalized = self.slang_dict.get(word, word)
            
            # Loại stopwords nếu cần
            if self.remove_stopwords and normalized in self.stop_words:
                continue
            
            normalized_words.append(normalized)
        
        return ' '.join(normalized_words)
    
    def preprocess(self, text: str) -> str:
        """
        Pipeline tiền xử lý đầy đủ
        """
        text = self.clean_text(text)
        text = self.normalize_slang(text)
        return text
    
    def preprocess_batch(self, texts: List[str]) -> List[str]:
        """
        Tiền xử lý hàng loạt
        """
        return [self.preprocess(text) for text in texts]


def load_and_split_data(
    train_csv_path: str,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load dữ liệu từ train.csv và chia thành train/val/test
    
    Args:
        train_csv_path: Đường dẫn tới train.csv
        test_size: Tỷ lệ test (0.2 = 20%)
        val_size: Tỷ lệ validation (0.1 = 10% của phần còn lại sau test)
        random_state: Random seed
    
    Returns:
        (train_df, val_df, test_df)
    """
    from sklearn.model_selection import train_test_split
    
    # Load data
    df = pd.read_csv(train_csv_path)
    
    # Giả định cột văn bản là 'comment_text' và 6 nhãn
    label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    
    # Kiểm tra cột
    assert 'comment_text' in df.columns, "Missing 'comment_text' column"
    for col in label_cols:
        assert col in df.columns, f"Missing label column: {col}"
    
    # Chia train/test trước
    train_val_df, test_df = train_test_split(
        df, 
        test_size=test_size, 
        random_state=random_state,
        stratify=None  # Bỏ stratify để tránh lỗi khi có combination nhãn hiếm
    )
    
    # Chia train/val
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_size,
        random_state=random_state,
        stratify=None  # Bỏ stratify để tránh lỗi khi có combination nhãn hiếm
    )
    
    print(f"✓ Data split:")
    print(f"  - Train: {len(train_df)} samples ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  - Val:   {len(val_df)} samples ({len(val_df)/len(df)*100:.1f}%)")
    print(f"  - Test:  {len(test_df)} samples ({len(test_df)/len(df)*100:.1f}%)")
    
    return train_df, val_df, test_df


def prepare_sequences(
    texts: List[str],
    tokenizer,
    max_len: int = 250
) -> np.ndarray:
    """
    Chuyển văn bản thành sequences và padding
    
    Args:
        texts: List văn bản
        tokenizer: Keras Tokenizer đã fit
        max_len: Độ dài tối đa của sequence
    
    Returns:
        Padded sequences (numpy array)
    """
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    
    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')
    
    return padded


if __name__ == "__main__":
    # Test preprocess
    preprocessor = TextPreprocessor(
        slang_dict_path="../Data/slang.csv",
        remove_stopwords=False
    )
    
    sample_texts = [
        "OMG this is so fking toxic!!! @user http://spam.com 😡",
        "You're such an idiot lol",
        "This is a normal comment."
    ]
    
    print("\n=== Text Preprocessing Test ===")
    for text in sample_texts:
        cleaned = preprocessor.preprocess(text)
        print(f"Original: {text}")
        print(f"Cleaned:  {cleaned}\n")

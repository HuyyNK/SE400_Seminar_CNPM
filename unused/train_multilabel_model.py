# train_multilabel_model_v4.py

import pandas as pd
import joblib
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from hybrid_classifier import clean_text, preprocess_text

def train_and_save_multilabel_classifier_v4():
    """
    Huấn luyện và lưu bộ phân loại đa nhãn phiên bản 4 (Undersampling):
    - Không dùng SMOTE hay class_weight.
    - Áp dụng Random Undersampling để tạo tập huấn luyện cân bằng hơn.
    - Sử dụng Logistic Regression.
    """
    # 1. Tải dữ liệu
    print("Đang tải dữ liệu train.csv...")
    data_path = Path(__file__).parent / 'Data' / 'train.csv'
    df = pd.read_csv(data_path).dropna(subset=['comment_text'])

    # 2. Tiền xử lý văn bản
    print("Đang tiền xử lý văn bản (bước này có thể mất vài phút)...")
    df['processed_comment'] = df['comment_text'].apply(lambda x: preprocess_text(clean_text(x)))
    
    label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    
    # 3. Huấn luyện một mô hình cho mỗi nhãn bằng phương pháp Undersampling
    classifiers = {}
    print("Bắt đầu huấn luyện 6 mô hình riêng biệt (phiên bản V4 - Undersampling)...")

    for i, label in enumerate(label_cols):
        print(f"\n  ({i+1}/6) Đang huấn luyện cho nhãn: '{label}'...")
        
        # Tách dữ liệu thành 2 lớp: positive (1) và negative (0)
        df_positive = df[df[label] == 1]
        df_negative = df[df[label] == 0]
        
        # Tính toán số lượng mẫu negative cần lấy
        # Chúng ta sẽ lấy số mẫu negative nhiều gấp 5 lần số mẫu positive
        # Điều này giúp giữ lại nhiều thông tin của lớp đa số hơn là cân bằng 1:1
        # Bạn có thể thay đổi tỷ lệ này (ví dụ: 3, 5, 10)
        negative_sample_size = min(len(df_negative), len(df_positive) * 5)
        
        print(f"    - Số mẫu Positive (1): {len(df_positive)}")
        print(f"    - Số mẫu Negative (0) sẽ được lấy: {negative_sample_size} (từ {len(df_negative)} mẫu)")

        # Lấy mẫu ngẫu nhiên từ lớp negative
        df_negative_sampled = df_negative.sample(n=negative_sample_size, random_state=42)
        
        # Kết hợp lại để tạo thành tập dữ liệu huấn luyện mới cho nhãn này
        df_train_balanced = pd.concat([df_positive, df_negative_sampled])
        
        # Xáo trộn dữ liệu
        df_train_balanced = df_train_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
        
        print(f"    - Tổng số mẫu huấn luyện cho nhãn '{label}': {len(df_train_balanced)}")

        X_train = df_train_balanced['processed_comment']
        y_train = df_train_balanced[label]
        
        # Tạo pipeline huấn luyện
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
            ('clf', LogisticRegression(solver='liblinear', random_state=42)) # Không dùng class_weight
        ])
        
        # Huấn luyện pipeline trên tập dữ liệu đã được undersample
        pipeline.fit(X_train, y_train)
        classifiers[label] = pipeline

    # 4. Lưu bộ phân loại
    print("\nĐã huấn luyện xong. Đang lưu mô hình V4...")
    save_dir = Path(__file__).parent / 'saved_models'
    save_dir.mkdir(exist_ok=True)
    joblib.dump(classifiers, save_dir / 'multilabel_classifiers.pkl')
    print(f"✓ Bộ phân loại đa nhãn V4 đã được lưu vào: {save_dir / 'multilabel_classifiers.pkl'}")

if __name__ == "__main__":
    train_and_save_multilabel_classifier_v4()
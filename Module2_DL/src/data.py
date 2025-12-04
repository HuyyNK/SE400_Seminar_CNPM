"""
Module 2: Utilities cho data loading và xử lý
"""

import os
import pandas as pd
import numpy as np
from typing import Tuple, Dict


def get_label_distribution(df: pd.DataFrame, label_cols: list) -> Dict:
    """
    Phân tích phân phối nhãn trong dataset
    
    Args:
        df: DataFrame chứa labels
        label_cols: List tên các cột nhãn
    
    Returns:
        Dict chứa thống kê
    """
    stats = {}
    
    for col in label_cols:
        stats[col] = {
            'count': int(df[col].sum()),
            'percentage': float(df[col].mean() * 100)
        }
    
    # Tính số lượng mẫu không độc hại
    clean_count = int((df[label_cols].sum(axis=1) == 0).sum())
    stats['clean'] = {
        'count': clean_count,
        'percentage': float(clean_count / len(df) * 100)
    }
    
    return stats


def print_label_distribution(df: pd.DataFrame, label_cols: list, title: str = "Label Distribution"):
    """
    In phân phối nhãn
    """
    stats = get_label_distribution(df, label_cols)
    
    print(f"\n{title}")
    print("=" * 50)
    for label, data in stats.items():
        print(f"{label:15} | Count: {data['count']:6} | {data['percentage']:5.2f}%")
    print("=" * 50)


def calculate_sample_weights(y: np.ndarray) -> np.ndarray:
    """
    Tính sample weights cho training (dựa trên số nhãn mỗi sample có)
    
    Args:
        y: Labels array (n_samples, n_labels)
    
    Returns:
        Sample weights array (n_samples,)
    """
    # Số nhãn positive mỗi sample
    num_labels = y.sum(axis=1)
    
    # Sample có nhiều nhãn được weight cao hơn
    weights = np.where(num_labels > 0, num_labels, 0.5)
    
    return weights


if __name__ == "__main__":
    # Test utilities
    train_csv = "../Data/train.csv"
    
    if os.path.exists(train_csv):
        df = pd.read_csv(train_csv)
        label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        
        print_label_distribution(df, label_cols)
    else:
        print(f"File not found: {train_csv}")

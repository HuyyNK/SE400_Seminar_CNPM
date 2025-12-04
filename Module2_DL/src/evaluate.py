"""
Module 2: Script đánh giá mô hình
- Tính AUC, Precision, Recall, F1 cho từng label
- Tìm ngưỡng tối ưu cho từng label
- Vẽ PR curves và confusion matrices
- Lưu báo cáo JSON
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, auc,
    f1_score, precision_score, recall_score,
    confusion_matrix, classification_report
)
from tensorflow import keras
from tensorflow.keras.preprocessing.text import tokenizer_from_json

# Import từ các module khác
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from preprocess import TextPreprocessor, load_and_split_data, prepare_sequences


def find_optimal_thresholds(y_true, y_pred, metric='f1'):
    """
    Tìm ngưỡng tối ưu cho từng label dựa trên F1 score hoặc Youden's index
    
    Args:
        y_true: Ground truth labels (n_samples, n_labels)
        y_pred: Predicted probabilities (n_samples, n_labels)
        metric: 'f1' hoặc 'youden'
    
    Returns:
        optimal_thresholds: List ngưỡng tối ưu cho từng label
    """
    n_labels = y_true.shape[1]
    optimal_thresholds = []
    
    for i in range(n_labels):
        precision, recall, thresholds = precision_recall_curve(y_true[:, i], y_pred[:, i])
        
        if metric == 'f1':
            # F1 = 2 * (precision * recall) / (precision + recall)
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
            optimal_idx = np.argmax(f1_scores)
            if optimal_idx < len(thresholds):
                optimal_threshold = thresholds[optimal_idx]
            else:
                optimal_threshold = 0.5
        elif metric == 'youden':
            # Youden's J = Sensitivity + Specificity - 1
            # Đối với PR curve, ta dùng precision và recall
            youden = recall + precision - 1
            optimal_idx = np.argmax(youden)
            if optimal_idx < len(thresholds):
                optimal_threshold = thresholds[optimal_idx]
            else:
                optimal_threshold = 0.5
        else:
            optimal_threshold = 0.5
        
        optimal_thresholds.append(optimal_threshold)
    
    return optimal_thresholds


def evaluate_at_thresholds(y_true, y_pred, thresholds, label_names):
    """
    Đánh giá performance ở các ngưỡng cho trước
    
    Args:
        y_true: Ground truth
        y_pred: Predicted probabilities
        thresholds: List ngưỡng cho từng label
        label_names: List tên labels
    
    Returns:
        metrics_dict: Dict chứa metrics cho từng label
    """
    metrics_dict = {}
    
    for i, label in enumerate(label_names):
        threshold = thresholds[i]
        y_pred_binary = (y_pred[:, i] >= threshold).astype(int)
        
        metrics_dict[label] = {
            'threshold': float(threshold),
            'auc': float(roc_auc_score(y_true[:, i], y_pred[:, i])),
            'precision': float(precision_score(y_true[:, i], y_pred_binary, zero_division=0)),
            'recall': float(recall_score(y_true[:, i], y_pred_binary, zero_division=0)),
            'f1': float(f1_score(y_true[:, i], y_pred_binary, zero_division=0))
        }
    
    # Macro averages
    metrics_dict['macro_avg'] = {
        'auc': float(np.mean([m['auc'] for m in metrics_dict.values() if isinstance(m, dict)])),
        'precision': float(np.mean([m['precision'] for m in metrics_dict.values() if isinstance(m, dict)])),
        'recall': float(np.mean([m['recall'] for m in metrics_dict.values() if isinstance(m, dict)])),
        'f1': float(np.mean([m['f1'] for m in metrics_dict.values() if isinstance(m, dict)]))
    }
    
    # Micro averages (flatten tất cả labels)
    all_thresholds = np.repeat(thresholds, y_true.shape[0]).reshape(y_true.shape)
    y_pred_binary_all = (y_pred >= all_thresholds).astype(int)
    
    metrics_dict['micro_avg'] = {
        'precision': float(precision_score(y_true.flatten(), y_pred_binary_all.flatten(), zero_division=0)),
        'recall': float(recall_score(y_true.flatten(), y_pred_binary_all.flatten(), zero_division=0)),
        'f1': float(f1_score(y_true.flatten(), y_pred_binary_all.flatten(), zero_division=0))
    }
    
    return metrics_dict


def plot_pr_curves(y_true, y_pred, label_names, output_path):
    """
    Vẽ Precision-Recall curves cho tất cả labels
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, label in enumerate(label_names):
        precision, recall, _ = precision_recall_curve(y_true[:, i], y_pred[:, i])
        pr_auc = auc(recall, precision)
        
        axes[i].plot(recall, precision, label=f'AUC = {pr_auc:.3f}')
        axes[i].set_xlabel('Recall')
        axes[i].set_ylabel('Precision')
        axes[i].set_title(f'PR Curve: {label}')
        axes[i].legend()
        axes[i].grid(True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved PR curves to {output_path}")


def evaluate_model(
    model_path: str,
    config_path: str,
    tokenizer_path: str,
    train_csv: str = '../Data/train.csv',
    slang_csv: str = '../Data/slang.csv',
    output_dir: str = '../artifacts'
):
    """
    Đánh giá mô hình trên validation và test set
    
    Args:
        model_path: Đường dẫn tới model .h5
        config_path: Đường dẫn tới config JSON
        tokenizer_path: Đường dẫn tới tokenizer JSON
        train_csv: Đường dẫn tới train.csv
        slang_csv: Đường dẫn tới slang.csv
        output_dir: Thư mục lưu kết quả
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print("Evaluating Model")
    print("="*60)
    
    # 1. Load config
    print("\n[1/6] Loading config...")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    max_len = config['max_len']
    label_cols = config['label_cols']
    model_type = config['model_type']
    
    # 2. Load tokenizer
    print("\n[2/6] Loading tokenizer...")
    with open(tokenizer_path, 'r', encoding='utf-8') as f:
        tokenizer_json = f.read()
    tokenizer = tokenizer_from_json(tokenizer_json)
    
    # 3. Load model
    print("\n[3/6] Loading model...")
    model = keras.models.load_model(model_path)
    model.summary()
    
    # 4. Load và tiền xử lý data
    print("\n[4/6] Loading and preprocessing data...")
    train_df, val_df, test_df = load_and_split_data(
        train_csv_path=train_csv,
        test_size=0.2,
        val_size=0.1
    )
    
    preprocessor = TextPreprocessor(slang_dict_path=slang_csv, remove_stopwords=False)
    
    val_texts = preprocessor.preprocess_batch(val_df['comment_text'].tolist())
    test_texts = preprocessor.preprocess_batch(test_df['comment_text'].tolist())
    
    X_val = prepare_sequences(val_texts, tokenizer, max_len)
    X_test = prepare_sequences(test_texts, tokenizer, max_len)
    
    y_val = val_df[label_cols].values
    y_test = test_df[label_cols].values
    
    # 5. Predict
    print("\n[5/6] Making predictions...")
    y_val_pred = model.predict(X_val, batch_size=512, verbose=1)
    y_test_pred = model.predict(X_test, batch_size=512, verbose=1)
    
    # 6. Evaluate
    print("\n[6/6] Evaluating and generating report...")
    
    # Tìm ngưỡng tối ưu trên validation set
    print("\n  Finding optimal thresholds on validation set...")
    optimal_thresholds = find_optimal_thresholds(y_val, y_val_pred, metric='f1')
    
    print("\n  Optimal Thresholds:")
    for i, (label, thresh) in enumerate(zip(label_cols, optimal_thresholds)):
        print(f"    {label}: {thresh:.3f}")
    
    # Đánh giá trên validation set
    print("\n  Evaluating on validation set...")
    val_metrics = evaluate_at_thresholds(y_val, y_val_pred, optimal_thresholds, label_cols)
    
    # Đánh giá trên test set
    print("\n  Evaluating on test set...")
    test_metrics = evaluate_at_thresholds(y_test, y_test_pred, optimal_thresholds, label_cols)
    
    # In kết quả
    print("\n" + "="*60)
    print("VALIDATION SET RESULTS")
    print("="*60)
    for label in label_cols:
        m = val_metrics[label]
        print(f"{label:15} | AUC: {m['auc']:.4f} | P: {m['precision']:.4f} | R: {m['recall']:.4f} | F1: {m['f1']:.4f}")
    print(f"\nMacro Avg      | AUC: {val_metrics['macro_avg']['auc']:.4f} | "
          f"P: {val_metrics['macro_avg']['precision']:.4f} | "
          f"R: {val_metrics['macro_avg']['recall']:.4f} | "
          f"F1: {val_metrics['macro_avg']['f1']:.4f}")
    
    print("\n" + "="*60)
    print("TEST SET RESULTS")
    print("="*60)
    for label in label_cols:
        m = test_metrics[label]
        print(f"{label:15} | AUC: {m['auc']:.4f} | P: {m['precision']:.4f} | R: {m['recall']:.4f} | F1: {m['f1']:.4f}")
    print(f"\nMacro Avg      | AUC: {test_metrics['macro_avg']['auc']:.4f} | "
          f"P: {test_metrics['macro_avg']['precision']:.4f} | "
          f"R: {test_metrics['macro_avg']['recall']:.4f} | "
          f"F1: {test_metrics['macro_avg']['f1']:.4f}")
    
    # Lưu báo cáo JSON
    report = {
        'model_type': model_type,
        'optimal_thresholds': {label: float(t) for label, t in zip(label_cols, optimal_thresholds)},
        'validation_metrics': val_metrics,
        'test_metrics': test_metrics
    }
    
    report_path = os.path.join(output_dir, f'report_baseline_dl_{model_type}.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    print(f"\n✓ Saved evaluation report to {report_path}")
    
    # Vẽ PR curves
    pr_curve_path = os.path.join(output_dir, f'pr_curves_{model_type}.png')
    plot_pr_curves(y_test, y_test_pred, label_cols, pr_curve_path)
    
    print("\n" + "="*60)
    print("Evaluation Complete!")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate trained model')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to model .h5 file')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config JSON')
    parser.add_argument('--tokenizer', type=str, required=True,
                        help='Path to tokenizer JSON')
    parser.add_argument('--train_csv', type=str, default='../Data/train.csv',
                        help='Path to train.csv')
    parser.add_argument('--slang_csv', type=str, default='../Data/slang.csv',
                        help='Path to slang.csv')
    parser.add_argument('--output', type=str, default='../artifacts',
                        help='Output directory')
    
    args = parser.parse_args()
    
    evaluate_model(
        model_path=args.model,
        config_path=args.config,
        tokenizer_path=args.tokenizer,
        train_csv=args.train_csv,
        slang_csv=args.slang_csv,
        output_dir=args.output
    )

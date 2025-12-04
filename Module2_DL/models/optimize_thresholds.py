#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
THRESHOLD OPTIMIZATION SCRIPT
==============================

Finds optimal per-label thresholds to maximize F1 score on validation set.

Usage:
    python optimize_thresholds.py
"""

import sys
import os
from pathlib import Path

# ============================================================================
# FIX 1: Windows Unicode Encoding
# ============================================================================
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# ============================================================================
# SETUP: Paths & Imports
# ============================================================================
module2_root = Path(__file__).parent.parent
sys.path.insert(0, str(module2_root))

import numpy as np
import pandas as pd
from pathlib import Path
import json
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from src.utils import LABEL_COLS, ARTIFACTS_DIR, REPORTS_DIR

print("\n" + "=" * 80)
print("THRESHOLD OPTIMIZATION")
print("=" * 80)
print(f"\nWorking directory: {module2_root}")
print(f"REPORTS_DIR: {REPORTS_DIR}")
print(f"ARTIFACTS_DIR: {ARTIFACTS_DIR}")

# ============================================================================
# STEP 1: Load Predictions and Ground Truth
# ============================================================================
print("\nSTEP 1: Load predictions and validation labels")
print("-" * 80)

# Load raw predictions
pred_path = REPORTS_DIR / "val_predictions.npy"
if not pred_path.exists():
    print(f"[X] Predictions file not found: {pred_path}")
    sys.exit(1)

predictions = np.load(pred_path)
print(f"[✓] Predictions loaded: {predictions.shape}")
print(f"    Range: [{predictions.min():.4f}, {predictions.max():.4f}]")

# Load validation labels
val_path = ARTIFACTS_DIR / "val_final.csv"
val_df = pd.read_csv(val_path)
y_true = val_df[LABEL_COLS].values
print(f"[✓] Labels loaded: {y_true.shape}")
print(f"    Labels: {LABEL_COLS}")

# ============================================================================
# STEP 2: Grid Search for Optimal Thresholds
# ============================================================================
print("\n" + "=" * 80)
print("STEP 2: Grid Search for Optimal Thresholds")
print("=" * 80)

thresholds_to_test = np.arange(0.1, 0.95, 0.05)
optimal_thresholds = {}
best_f1_scores = {}

print(f"\nSearching thresholds: {[f'{t:.2f}' for t in thresholds_to_test]}")
print()

for label_idx, label_name in enumerate(LABEL_COLS):
    y_label_true = y_true[:, label_idx]
    y_label_pred = predictions[:, label_idx]
    
    best_threshold = 0.5
    best_f1 = 0.0
    
    for threshold in thresholds_to_test:
        y_label_pred_binary = (y_label_pred >= threshold).astype(int)
        f1 = f1_score(y_label_true, y_label_pred_binary, zero_division=0)
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    optimal_thresholds[label_name] = float(best_threshold)
    best_f1_scores[label_name] = float(best_f1)
    
    print(f"  {label_name:20s} → threshold = {best_threshold:.2f}, F1 = {best_f1:.4f}")

print()

# ============================================================================
# STEP 3: Compare Static vs Optimized Thresholds
# ============================================================================
print("=" * 80)
print("STEP 3: Compare Static (0.5) vs Optimized Thresholds")
print("=" * 80)

def evaluate_with_thresholds(predictions, y_true, thresholds_dict, label_cols):
    """Evaluate predictions with given thresholds"""
    metrics = {}
    
    for label_idx, label_name in enumerate(label_cols):
        y_label_true = y_true[:, label_idx]
        y_label_pred = predictions[:, label_idx]
        
        threshold = thresholds_dict[label_name]
        y_label_pred_binary = (y_label_pred >= threshold).astype(int)
        
        precision = precision_score(y_label_true, y_label_pred_binary, zero_division=0)
        recall = recall_score(y_label_true, y_label_pred_binary, zero_division=0)
        f1 = f1_score(y_label_true, y_label_pred_binary, zero_division=0)
        auc = roc_auc_score(y_label_true, y_label_pred)
        
        metrics[label_name] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'threshold': threshold
        }
    
    return metrics

# Static thresholds (all 0.5)
static_thresholds = {label: 0.5 for label in LABEL_COLS}
static_metrics = evaluate_with_thresholds(predictions, y_true, static_thresholds, LABEL_COLS)

# Optimized thresholds
optimized_metrics = evaluate_with_thresholds(predictions, y_true, optimal_thresholds, LABEL_COLS)

# Print comparison
print("\n" + "STATIC THRESHOLDS (0.5)".center(90))
print("-" * 90)
print(f"{'Label':<20} {'Threshold':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'AUC':<12}")
print("-" * 90)

static_f1_sum = 0
for label in LABEL_COLS:
    m = static_metrics[label]
    print(f"{label:<20} {m['threshold']:<12.2f} {m['precision']:<12.4f} {m['recall']:<12.4f} {m['f1']:<12.4f} {m['auc']:<12.4f}")
    static_f1_sum += m['f1']

static_f1_macro = static_f1_sum / len(LABEL_COLS)
print("-" * 90)
print(f"{'MACRO AVERAGE':<20} {'-':<12} {'-':<12} {'-':<12} {static_f1_macro:<12.4f} {'-':<12}")

# Optimized metrics
print("\n" + "OPTIMIZED THRESHOLDS".center(90))
print("-" * 90)
print(f"{'Label':<20} {'Threshold':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'AUC':<12}")
print("-" * 90)

optimized_f1_sum = 0
for label in LABEL_COLS:
    m = optimized_metrics[label]
    print(f"{label:<20} {m['threshold']:<12.2f} {m['precision']:<12.4f} {m['recall']:<12.4f} {m['f1']:<12.4f} {m['auc']:<12.4f}")
    optimized_f1_sum += m['f1']

optimized_f1_macro = optimized_f1_sum / len(LABEL_COLS)
print("-" * 90)
print(f"{'MACRO AVERAGE':<20} {'-':<12} {'-':<12} {'-':<12} {optimized_f1_macro:<12.4f} {'-':<12}")

# ============================================================================
# STEP 4: Calculate Improvement
# ============================================================================
print("\n" + "=" * 80)
print("STEP 4: Improvement Summary")
print("=" * 80)

improvement = optimized_f1_macro - static_f1_macro
improvement_pct = (improvement / static_f1_macro * 100) if static_f1_macro > 0 else 0

print(f"\nF1 Score Improvement:")
print(f"  Static (0.5):   {static_f1_macro:.4f}")
print(f"  Optimized:      {optimized_f1_macro:.4f}")
print(f"  Improvement:    +{improvement:.4f} ({improvement_pct:+.1f}%)")

# ============================================================================
# STEP 5: Save Optimized Thresholds
# ============================================================================
print("\n" + "=" * 80)
print("STEP 5: Save Optimized Thresholds")
print("=" * 80)

# Create output directory if needed
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Save thresholds
thresholds_output = {
    'static': static_metrics,
    'optimized': optimized_metrics,
    'optimal_thresholds': optimal_thresholds,
    'improvement': {
        'static_f1_macro': static_f1_macro,
        'optimized_f1_macro': optimized_f1_macro,
        'improvement': improvement,
        'improvement_pct': improvement_pct
    }
}

thresholds_path = REPORTS_DIR / "optimized_thresholds.json"
with open(thresholds_path, 'w', encoding='utf-8') as f:
    json.dump(thresholds_output, f, indent=2)

print(f"\n[✓] Thresholds saved to: {thresholds_path}")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("OPTIMIZATION COMPLETE!")
print("=" * 80)

print(f"\nOptimal Thresholds (per-label):")
for label, threshold in optimal_thresholds.items():
    print(f"  {label:<20} = {threshold:.2f}")

print(f"\nPerformance Boost:")
print(f"  Macro F1: {static_f1_macro:.4f} → {optimized_f1_macro:.4f} (+{improvement:.4f})")

print("\nOutput Files:")
print(f"  {thresholds_path}")

print("\n" + "=" * 80)

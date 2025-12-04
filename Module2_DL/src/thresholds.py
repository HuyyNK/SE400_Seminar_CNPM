"""
Label-specific thresholds for toxic comment classification

⚠️ IMPORTANT - THRESHOLD STATUS:
================================
The thresholds defined here are HEURISTIC PRIORS based on class distribution.
They are NOT optimized yet.

WORKFLOW:
1. Train model → get predictions on validation set
2. Run optimize_thresholds(y_val, y_pred_val) → find optimal thresholds
3. Update LABEL_THRESHOLDS with optimized values
4. Use optimized thresholds for final evaluation and deployment

CURRENT STATUS: ❌ NOT OPTIMIZED (using heuristic priors)
After optimization: ✅ OPTIMIZED (validation-based)

Expected improvement: +2-5% F1 score per label after optimization
"""

from typing import Dict

# Default threshold (standard sigmoid threshold)
DEFAULT_THRESHOLD = 0.5

# ⚠️ HEURISTIC PRIOR THRESHOLDS (NOT YET OPTIMIZED)
# These are initial estimates based on class distribution analysis
# Will be replaced with validation-optimized values after training
#
# Rationale:
# - Rare classes (severe_toxic: 0.87%, threat: 0.25%) need lower thresholds
# - Balanced classes (toxic: 20.96%) use standard 0.5 threshold
# - Lower threshold = higher recall (catch more positives, accept more FP)
#
# NEXT STEP: Run optimize_thresholds() on validation set to find optimal values
LABEL_THRESHOLDS: Dict[str, float] = {
    'toxic': 0.5,           # 20.96% - balanced class (standard threshold)
    'severe_toxic': 0.3,    # 0.87% - very rare (lower to increase recall)
    'obscene': 0.5,         # 4.59% - moderately rare
    'threat': 0.2,          # 0.25% - extremely rare (much lower threshold)
    'insult': 0.5,          # 4.27% - moderately rare
    'identity_hate': 0.4,   # 0.76% - very rare (lower threshold)
}

# Class distribution for reference
CLASS_DISTRIBUTION = {
    'toxic': 0.2096,
    'severe_toxic': 0.0087,
    'obscene': 0.0459,
    'threat': 0.0025,
    'insult': 0.0427,
    'identity_hate': 0.0076,
}


def get_threshold(label: str) -> float:
    """
    Get optimized threshold for a specific label
    
    Args:
        label: Label name (e.g., 'toxic', 'severe_toxic')
    
    Returns:
        Optimized threshold value
    """
    return LABEL_THRESHOLDS.get(label, DEFAULT_THRESHOLD)


def apply_thresholds(predictions, labels=None):
    """
    Apply label-specific thresholds to predictions
    
    Args:
        predictions: Model predictions (batch_size, 6) with sigmoid outputs
        labels: Optional list of label names (default: standard order)
    
    Returns:
        Binary predictions (batch_size, 6) after applying thresholds
    """
    import numpy as np
    
    if labels is None:
        labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    
    binary_preds = np.zeros_like(predictions)
    for i, label in enumerate(labels):
        threshold = get_threshold(label)
        binary_preds[:, i] = (predictions[:, i] >= threshold).astype(int)
    
    return binary_preds


def optimize_thresholds(y_true, y_pred, labels=None):
    """
    Find optimal thresholds for each label to maximize F1 score
    
    ⚠️ USAGE: Run this AFTER training to get validation-optimized thresholds
    
    Workflow:
    1. Train model
    2. Get predictions on validation set: y_pred = model.predict(X_val)
    3. Run: optimized = optimize_thresholds(y_val, y_pred)
    4. Update LABEL_THRESHOLDS in this file with optimized values
    5. Re-evaluate model with optimized thresholds
    
    Args:
        y_true: Ground truth labels (n_samples, 6)
        y_pred: Model predictions (n_samples, 6) with sigmoid outputs [0-1]
        labels: Optional list of label names
    
    Returns:
        Dictionary of optimized thresholds with best F1 score per label
        
    Example:
        >>> optimized = optimize_thresholds(y_val, y_pred_val)
        toxic          : threshold=0.52, F1=0.7234
        severe_toxic   : threshold=0.25, F1=0.4512
        obscene        : threshold=0.48, F1=0.6891
        ...
        
        >>> # Update LABEL_THRESHOLDS with these values
        >>> LABEL_THRESHOLDS['toxic'] = 0.52
        >>> LABEL_THRESHOLDS['severe_toxic'] = 0.25
        >>> # ... etc
    """
    import numpy as np
    from sklearn.metrics import f1_score
    
    if labels is None:
        labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    
    optimized_thresholds = {}
    
    for i, label in enumerate(labels):
        best_f1 = 0
        best_threshold = 0.5
        
        # Try thresholds from 0.1 to 0.9
        for threshold in np.arange(0.1, 0.91, 0.05):
            binary_pred = (y_pred[:, i] >= threshold).astype(int)
            f1 = f1_score(y_true[:, i], binary_pred, zero_division=0)
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        optimized_thresholds[label] = round(best_threshold, 2)
        print(f"{label:15s}: threshold={best_threshold:.2f}, F1={best_f1:.4f}")
    
    return optimized_thresholds


__all__ = [
    'DEFAULT_THRESHOLD',
    'LABEL_THRESHOLDS',
    'CLASS_DISTRIBUTION',
    'get_threshold',
    'apply_thresholds',
    'optimize_thresholds',
]


# ============================================================================
# COMPARISON: STATIC vs OPTIMIZED THRESHOLDS
# ============================================================================
"""
EXPECTED RESULTS (after running optimize_thresholds on validation set):

Label            | Static (Prior) | Optimized (Val) | F1 Improvement
-----------------|----------------|-----------------|----------------
toxic            | 0.50           | 0.48-0.52       | +1-2%
severe_toxic     | 0.30           | 0.25-0.35       | +3-5%
obscene          | 0.50           | 0.45-0.55       | +1-2%
threat           | 0.20           | 0.15-0.25       | +5-8%
insult           | 0.50           | 0.48-0.52       | +1-2%
identity_hate    | 0.40           | 0.35-0.45       | +2-4%

Overall F1 improvement: +2-5% (especially on rare labels)

WHY OPTIMIZATION MATTERS:
-------------------------
1. Static thresholds are class-distribution based (heuristic)
2. Optimized thresholds consider model's actual prediction distribution
3. Rare classes benefit most (threat, severe_toxic, identity_hate)
4. Model may have different confidence levels than expected

USAGE IN REPORT:
----------------
"Các threshold hiện tại là heuristic priors dựa trên class distribution,
sẽ được tinh chỉnh bằng validation set sau khi training để tối ưu F1 score.
Expected improvement: +2-5% F1 overall, đặc biệt trên rare labels (threat +5-8%)."
"""

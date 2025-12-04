"""
Data Preparation Script - Simplified Version
Combines Kaggle + Twitter datasets without augmentation

Usage:
    python prepare_data.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import sys
import os

# Define paths directly (no imports to avoid circular dependencies)
PROJECT_ROOT = Path(__file__).parent
WORKSPACE_ROOT = PROJECT_ROOT.parent
DATA_ROOT = WORKSPACE_ROOT / 'Data'

KAGGLE_TRAIN_CSV = DATA_ROOT / 'train.csv'
TWITTER_TRAIN_CSV = WORKSPACE_ROOT / 'labeled_clean.csv'
ARTIFACTS_DIR = PROJECT_ROOT / 'artifacts'

print("="*60)
print("📊 DATA PREPARATION - SIMPLIFIED")
print("="*60)

# Create artifacts directory
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
print(f"\n✅ Created artifacts directory: {ARTIFACTS_DIR}")

# Define output paths
TRAIN_FINAL_CSV = ARTIFACTS_DIR / 'train_final.csv'
VAL_FINAL_CSV = ARTIFACTS_DIR / 'val_final.csv'
TRAIN_COMBINED_CSV = ARTIFACTS_DIR / 'train_combined_final.csv'

# Load datasets
print("\n[1/5] Loading datasets...")
print("-" * 60)

# Load Kaggle data
try:
    df_kaggle = pd.read_csv(str(KAGGLE_TRAIN_CSV))
    print(f"  ✓ Kaggle toxic comments: {len(df_kaggle):,} samples")
    print(f"    Columns: {list(df_kaggle.columns)}")
except FileNotFoundError:
    print(f"  ✗ ERROR: Kaggle data not found at {KAGGLE_TRAIN_CSV}")
    sys.exit(1)

# Load Twitter data
try:
    df_twitter = pd.read_csv(str(TWITTER_TRAIN_CSV))
    print(f"  ✓ Twitter toxic tweets: {len(df_twitter):,} samples")
    print(f"    Columns: {list(df_twitter.columns)}")
except FileNotFoundError:
    print(f"  ✗ ERROR: Twitter data not found at {TWITTER_TRAIN_CSV}")
    sys.exit(1)

# Standardize columns
print("\n[2/5] Standardizing columns...")
print("-" * 60)

label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

# Process Kaggle data
if 'id' in df_kaggle.columns:
    df_kaggle = df_kaggle.drop(columns=['id'])

keep_cols = ['comment_text'] + label_cols
df_kaggle = df_kaggle[keep_cols].copy()
print(f"  ✓ Kaggle data: {len(df_kaggle):,} samples with {len(keep_cols)} columns")

# Process Twitter data
# Map 'class' column: 0=non-toxic, 1=toxic, 2=very toxic
if 'class' in df_twitter.columns:
    df_twitter['toxic'] = (df_twitter['class'] > 0).astype(int)
    print(f"    - Mapped 'class' to 'toxic': {df_twitter['toxic'].sum():,} toxic samples")

# Ensure all label columns exist (Twitter only has toxic, rest are 0)
for col in label_cols:
    if col not in df_twitter.columns:
        df_twitter[col] = 0

# Rename 'tweet' to 'comment_text'
if 'tweet' in df_twitter.columns:
    df_twitter = df_twitter.rename(columns={'tweet': 'comment_text'})

df_twitter = df_twitter[keep_cols].copy()
print(f"  ✓ Twitter data: {len(df_twitter):,} samples with {len(keep_cols)} columns")

# Combine datasets
print("\n[3/5] Combining datasets...")
print("-" * 60)

df_combined = pd.concat([df_kaggle, df_twitter], ignore_index=True)
print(f"  ✓ Combined: {len(df_combined):,} samples")

# Remove duplicates
original_len = len(df_combined)
df_combined = df_combined.drop_duplicates(subset=['comment_text'], keep='first')
duplicates_removed = original_len - len(df_combined)
print(f"  ✓ Removed {duplicates_removed:,} duplicates")
print(f"  ✓ Final: {len(df_combined):,} unique samples")

# Shuffle
df_combined = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)
print(f"  ✓ Shuffled with random_state=42")

# Label distribution
print("\n[4/5] Analyzing label distribution...")
print("-" * 60)

print(f"\n  Combined dataset ({len(df_combined):,} samples):")
print(f"  {'Label':<20} {'Count':>10} {'Percentage':>12}")
print(f"  {'-'*20} {'-'*10} {'-'*12}")

for label in label_cols:
    count = df_combined[label].sum()
    pct = count / len(df_combined) * 100
    print(f"  {label:<20} {count:>10,} {pct:>11.2f}%")

# Check multi-label samples
multi_label = (df_combined[label_cols].sum(axis=1) > 1).sum()
print(f"\n  Multi-label samples: {multi_label:,} ({multi_label/len(df_combined)*100:.2f}%)")

# Train/validation split
print(f"\n[5/5] Creating train/val split (85% / 15%)...")
print("-" * 60)

train_df, val_df = train_test_split(
    df_combined,
    test_size=0.15,
    random_state=42,
    stratify=df_combined['toxic']  # Stratify by toxic label
)

print(f"  ✓ Train set: {len(train_df):,} samples ({len(train_df)/len(df_combined)*100:.1f}%)")
print(f"  ✓ Validation set: {len(val_df):,} samples ({len(val_df)/len(df_combined)*100:.1f}%)")

# Show label distribution in splits
print("\n  Train set distribution:")
for label in label_cols:
    count = train_df[label].sum()
    pct = count / len(train_df) * 100
    print(f"    {label:<20} {count:>10,} ({pct:>5.2f}%)")

print("\n  Validation set distribution:")
for label in label_cols:
    count = val_df[label].sum()
    pct = count / len(val_df) * 100
    print(f"    {label:<20} {count:>10,} ({pct:>5.2f}%)")

# Save datasets
print(f"\n[6/6] Saving datasets...")
print("-" * 60)

# Save combined
df_combined.to_csv(str(TRAIN_COMBINED_CSV), index=False)
print(f"  ✓ Combined dataset saved: {TRAIN_COMBINED_CSV}")
print(f"    Size: {TRAIN_COMBINED_CSV.stat().st_size / 1024 / 1024:.2f} MB")

# Save train
train_df.to_csv(str(TRAIN_FINAL_CSV), index=False)
print(f"  ✓ Train dataset saved: {TRAIN_FINAL_CSV}")
print(f"    Size: {TRAIN_FINAL_CSV.stat().st_size / 1024 / 1024:.2f} MB")

# Save validation
val_df.to_csv(str(VAL_FINAL_CSV), index=False)
print(f"  ✓ Validation dataset saved: {VAL_FINAL_CSV}")
print(f"    Size: {VAL_FINAL_CSV.stat().st_size / 1024 / 1024:.2f} MB")

# Summary
print("\n" + "="*60)
print("✅ DATA PREPARATION COMPLETE")
print("="*60)

print(f"\n📊 Dataset Summary:")
print(f"  Total samples: {len(df_combined):,}")
print(f"    - Kaggle toxic comments: {len(df_kaggle):,}")
print(f"    - Twitter toxic tweets: {len(df_twitter):,}")
print(f"  Duplicates removed: {duplicates_removed:,}")
print(f"  Multi-label samples: {multi_label:,}")

print(f"\n📁 Files created:")
print(f"  1. {TRAIN_COMBINED_CSV.name}")
print(f"     → {len(df_combined):,} samples (all data combined)")
print(f"  2. {TRAIN_FINAL_CSV.name}")
print(f"     → {len(train_df):,} samples (85% for training)")
print(f"  3. {VAL_FINAL_CSV.name}")
print(f"     → {len(val_df):,} samples (15% for validation)")

print(f"\n🎯 Label Balance:")
toxic_ratio = df_combined['toxic'].sum() / len(df_combined) * 100
print(f"  Toxic: {toxic_ratio:.2f}% | Non-toxic: {100-toxic_ratio:.2f}%")

print(f"\n🚀 Next step: python train.py")
print("="*60)

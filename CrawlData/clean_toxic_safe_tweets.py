"""
Script to clean toxic "Safe" tweets from dataset
Remove or relabel Safe tweets that contain toxic words
"""

import pandas as pd
from pathlib import Path

# Load improved dataset
data_path = Path(__file__).parent.parent / 'Data' / 'labeled_clean_improved.csv'
df = pd.read_csv(data_path)

print("="*80)
print("CLEANING TOXIC 'SAFE' TWEETS")
print("="*80)

print(f"\nOriginal dataset size: {len(df)}")
print(f"Original class distribution:")
print(df['class'].value_counts().sort_index())

# Define toxic words
toxic_words = [
    'fuck', 'fucking', 'fucked', 'shit', 'bitch', 'bitches',
    'nigger', 'niggers', 'nigga', 'niggas', 'cunt', 'whore', 'slut',
    'bastard', 'asshole', 'retard', 'fag', 'faggot', 'dick', 
    'pussy', 'cock', 'kill', 'hate', 'die', 'death', 'stupid',
    'dumb', 'idiot', 'trash', 'worthless', 'scum'
]

# Function to check toxic words
def has_toxic(text):
    """Check if text contains any toxic words"""
    text_lower = str(text).lower()
    found_words = [word for word in toxic_words if word in text_lower]
    return len(found_words) > 0, found_words

# Find Safe tweets with toxic words
safe_tweets = df[df['class'] == 0].copy()
safe_tweets['has_toxic'], safe_tweets['toxic_words_found'] = zip(*safe_tweets['tweet'].apply(has_toxic))
toxic_safe = safe_tweets[safe_tweets['has_toxic']]

print(f"\n Found {len(toxic_safe)} Safe tweets containing toxic words ({len(toxic_safe)/len(safe_tweets)*100:.1f}%)")

# Show examples
print("\nExamples of toxic 'Safe' tweets to be removed:")
print("-"*80)
for idx, row in toxic_safe.head(10).iterrows():
    print(f"\n{row['tweet'][:100]}...")
    print(f"  Toxic words: {row['toxic_words_found']}")

# Option 1: REMOVE toxic "Safe" tweets (RECOMMENDED)
print("\n" + "="*80)
print("OPTION 1: REMOVE toxic Safe tweets")
print("="*80)

df_cleaned = df[~df.index.isin(toxic_safe.index)].copy()
print(f"\nDataset after removal:")
print(f"  Total: {len(df_cleaned)} (removed {len(toxic_safe)} tweets)")
print(f"  Class distribution:")
print(df_cleaned['class'].value_counts().sort_index())
print(f"\n  Percentages:")
print(df_cleaned['class'].value_counts(normalize=True).sort_index() * 100)

# Save cleaned dataset
output_path = Path(__file__).parent.parent / 'Data' / 'labeled_clean_fixed.csv'
df_cleaned.to_csv(output_path, index=False)
print(f"\n✓ Cleaned dataset saved to: {output_path}")

# Option 2: RELABEL toxic "Safe" tweets to Class 1 (Hate Speech)
print("\n" + "="*80)
print("OPTION 2: RELABEL toxic Safe tweets to Hate Speech")
print("="*80)

df_relabeled = df.copy()
df_relabeled.loc[toxic_safe.index, 'class'] = 1  # Relabel to Hate Speech

print(f"\nDataset after relabeling:")
print(f"  Total: {len(df_relabeled)}")
print(f"  Class distribution:")
print(df_relabeled['class'].value_counts().sort_index())
print(f"\n  Percentages:")
print(df_relabeled['class'].value_counts(normalize=True).sort_index() * 100)

# Save relabeled dataset
relabeled_path = Path(__file__).parent.parent / 'Data' / 'labeled_clean_relabeled_v2.csv'
df_relabeled.to_csv(relabeled_path, index=False)
print(f"\n✓ Relabeled dataset saved to: {relabeled_path}")

# Save toxic safe tweets for review
toxic_safe_path = Path(__file__).parent.parent / 'Data' / 'toxic_safe_tweets_removed.csv'
toxic_safe[['class', 'tweet', 'toxic_words_found']].to_csv(toxic_safe_path, index=False)
print(f"✓ Removed toxic Safe tweets saved to: {toxic_safe_path}")

# Statistics
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

print(f"\nORIGINAL DATASET PROBLEMS:")
print(f"  - Toxic 'Safe' tweets: {len(toxic_safe)} ({len(toxic_safe)/len(safe_tweets)*100:.1f}% of Safe class)")
print(f"  - This is why model predicted positive text as VIOLATION!")
print(f"  - Model learned: 'hate', 'nigger', 'fuck' = SAFE ❌")

print(f"\nFIXED DATASET (Option 1 - REMOVED):")
print(f"  - Safe: {len(df_cleaned[df_cleaned['class']==0])} ({len(df_cleaned[df_cleaned['class']==0])/len(df_cleaned)*100:.1f}%)")
print(f"  - Violation: {len(df_cleaned[df_cleaned['class']!=0])} ({len(df_cleaned[df_cleaned['class']!=0])/len(df_cleaned)*100:.1f}%)")
print(f"  - Ratio: {len(df_cleaned[df_cleaned['class']!=0]) / len(df_cleaned[df_cleaned['class']==0]):.1f}:1")

print(f"\nFIXED DATASET (Option 2 - RELABELED):")
print(f"  - Safe: {len(df_relabeled[df_relabeled['class']==0])} ({len(df_relabeled[df_relabeled['class']==0])/len(df_relabeled)*100:.1f}%)")
print(f"  - Violation: {len(df_relabeled[df_relabeled['class']!=0])} ({len(df_relabeled[df_relabeled['class']!=0])/len(df_relabeled)*100:.1f}%)")
print(f"  - Ratio: {len(df_relabeled[df_relabeled['class']!=0]) / len(df_relabeled[df_relabeled['class']==0]):.1f}:1")

print("\n" + "="*80)
print("RECOMMENDATION")
print("="*80)
print("\nUse: labeled_clean_fixed.csv (Option 1 - Removed toxic Safe)")
print("Why: Cleaner data, no contradicting labels")
print("\nSTILL NEED: Add 3,000-5,000 REAL Safe tweets for balance!")
print("Sources: Twitter positive hashtags, Reddit r/wholesome, positive news")

print("\n" + "="*80)
print("NEXT STEPS")
print("="*80)
print("1. Review toxic_safe_tweets_removed.csv to verify removal is correct")
print("2. Use labeled_clean_fixed.csv for training")
print("3. Re-run notebook from Section 3 (Load Data) onwards")
print("4. Collect REAL positive tweets to add to dataset")
print("5. Aim for 30% Safe, 70% Violation ratio")

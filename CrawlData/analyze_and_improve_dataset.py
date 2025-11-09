"""
Script to analyze and improve labeled_clean.csv dataset
- Check class distribution
- Find mislabeled examples
- Generate additional safe data
- Balance dataset
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path

# Load dataset
data_path = Path(__file__).parent.parent / 'Data' / 'labeled_clean.csv'
df = pd.read_csv(data_path)

print("="*80)
print("DATASET ANALYSIS")
print("="*80)

print(f"\nTotal rows: {len(df)}")
print(f"\nClass distribution:")
print(df['class'].value_counts().sort_index())
print(f"\nClass percentages:")
print(df['class'].value_counts(normalize=True).sort_index() * 100)

# Check for mislabeled examples
print("\n" + "="*80)
print("CHECKING FOR POTENTIAL MISLABELED DATA")
print("="*80)

# Positive words that should indicate SAFE content
positive_words = ['love', 'amazing', 'wonderful', 'great', 'excellent', 'beautiful', 
                  'fantastic', 'awesome', 'thank', 'thanks', 'grateful', 'appreciate',
                  'happy', 'joy', 'perfect', 'best', 'nice', 'good', 'well done']

# Toxic words that should indicate VIOLATION
toxic_words = ['fuck', 'shit', 'bitch', 'nigger', 'nigga', 'cunt', 'whore', 'slut',
               'bastard', 'asshole', 'retard', 'fag', 'dick', 'pussy', 'cock', 'kill',
               'hate', 'die', 'death', 'stupid', 'dumb', 'idiot', 'trash', 'worthless']

def check_words(text, word_list):
    """Check if any words from word_list appear in text"""
    text_lower = str(text).lower()
    found = [word for word in word_list if word in text_lower]
    return found

# Find violations with positive words (potential mislabeling)
violations = df[df['class'] != 0].copy()
violations['positive_words'] = violations['tweet'].apply(lambda x: check_words(x, positive_words))
violations['has_positive'] = violations['positive_words'].apply(lambda x: len(x) > 0)
violations['toxic_words'] = violations['tweet'].apply(lambda x: check_words(x, toxic_words))
violations['has_toxic'] = violations['toxic_words'].apply(lambda x: len(x) > 0)

# Suspicious: has positive words but NO toxic words
suspicious = violations[(violations['has_positive']) & (~violations['has_toxic'])]

print(f"\nSuspicious VIOLATION tweets (have positive words but NO toxic words): {len(suspicious)}")
print("\nTop 20 examples:")
for idx, row in suspicious.head(20).iterrows():
    print(f"\nClass {row['class']}: {row['tweet'][:120]}")
    print(f"  Positive words: {row['positive_words']}")

# Check SAFE tweets
safe_tweets = df[df['class'] == 0].copy()
safe_tweets['toxic_words'] = safe_tweets['tweet'].apply(lambda x: check_words(x, toxic_words))
safe_tweets['has_toxic'] = safe_tweets['toxic_words'].apply(lambda x: len(x) > 0)

suspicious_safe = safe_tweets[safe_tweets['has_toxic']]
print(f"\n\nSuspicious SAFE tweets (contain toxic words): {len(suspicious_safe)}")
if len(suspicious_safe) > 0:
    print("\nTop 10 examples:")
    for idx, row in suspicious_safe.head(10).iterrows():
        print(f"\nClass 0: {row['tweet'][:120]}")
        print(f"  Toxic words: {row['toxic_words']}")

# Sample current SAFE tweets
print("\n" + "="*80)
print("CURRENT SAFE TWEETS SAMPLE")
print("="*80)
for idx, row in df[df['class']==0].head(30).iterrows():
    print(f"\n{row['tweet'][:150]}")

# Statistics
print("\n" + "="*80)
print("RECOMMENDATIONS")
print("="*80)
print(f"\n1. Current Safe tweets: {len(safe_tweets)} (5.8%)")
print(f"   Recommended: At least {len(df) * 0.3:.0f} tweets (30% of dataset)")
print(f"   Need to add: {len(df) * 0.3 - len(safe_tweets):.0f} SAFE tweets")

print(f"\n2. Suspicious violations (may be mislabeled): {len(suspicious)}")
print(f"   These should be manually reviewed and potentially relabeled")

print(f"\n3. Current class imbalance ratio:")
print(f"   Class 1 (Hate): {len(df[df['class']==1])} ({len(df[df['class']==1])/len(df)*100:.1f}%)")
print(f"   Class 2 (Offensive): {len(df[df['class']==2])} ({len(df[df['class']==2])/len(df)*100:.1f}%)")
print(f"   Class 0 (Safe): {len(df[df['class']==0])} ({len(df[df['class']==0])/len(df)*100:.1f}%)")
print(f"   Imbalance ratio: {len(df[df['class']!=0]) / len(df[df['class']==0]):.1f}:1")

print("\n" + "="*80)
print("GENERATING IMPROVED DATASET")
print("="*80)

# Create improved dataset
improved_df = df.copy()

# Option 1: Relabel suspicious violations that are likely safe
# (Có positive words nhưng không có toxic words)
print(f"\nRelabeling {len(suspicious)} suspicious violations to SAFE...")
for idx in suspicious.index:
    improved_df.loc[idx, 'class'] = 0
    
print(f"After relabeling:")
print(improved_df['class'].value_counts().sort_index())

# Option 2: Generate safe tweets
print("\n\nGenerating additional SAFE tweets...")

# Common safe tweet patterns
safe_patterns = [
    "Good morning everyone! Have a great day!",
    "Thank you so much for your help today",
    "This is amazing work, congratulations!",
    "I appreciate all your support and kindness",
    "What a beautiful sunset tonight",
    "Having a wonderful time with family today",
    "Just finished a great book, highly recommend it",
    "Looking forward to the weekend!",
    "Happy birthday! Hope you have an amazing day",
    "Thanks for being such a good friend",
    "This coffee is perfect, exactly what I needed",
    "Great news everyone! The project is complete",
    "Feeling grateful for all the good things in life",
    "The weather is absolutely perfect today",
    "Congratulations on your achievement!",
    "Such a peaceful morning, love it",
    "Thank you for the lovely gift!",
    "This movie was fantastic, loved every minute",
    "Hope everyone is having a great day",
    "So proud of what we accomplished together",
    "Beautiful flowers in the garden today",
    "Just had the best meal ever!",
    "Thankful for good health and happiness",
    "Great presentation today, well done!",
    "Love spending time with my friends",
    "What a wonderful experience this has been",
    "Excited for the new opportunities ahead",
    "This music is so relaxing and beautiful",
    "Appreciate all the positive energy today",
    "Such a lovely surprise, thank you!",
    "Happy to help anytime you need",
    "This is going to be a great year",
    "Wonderful collaboration, thanks everyone",
    "Feeling blessed and grateful today",
    "Love the positive vibes here",
    "Great job team! We did it",
    "This is exactly what I was hoping for",
    "Thank you for making my day better",
    "Perfect timing, much appreciated",
    "So happy with how things turned out",
]

# Extend with more variations
additional_safe = []
templates = [
    "I really {} this {}!",
    "What a {} {} this is",
    "{} to see such {} work",
    "This is absolutely {}",
    "{} everyone for being so {}",
    "Having a {} time at {}",
    "The {} here is {}",
    "Just want to say {}",
    "So {} about this {}",
    "{} day with {} people",
]

positive_adjectives = ['amazing', 'wonderful', 'great', 'excellent', 'fantastic', 'awesome',
                       'beautiful', 'lovely', 'perfect', 'incredible', 'outstanding']
positive_nouns = ['day', 'moment', 'experience', 'time', 'event', 'opportunity', 'project',
                  'achievement', 'success', 'journey']
positive_verbs = ['love', 'enjoy', 'appreciate', 'admire', 'value']
positive_phrases = ['thank you', 'well done', 'good job', 'happy', 'grateful', 'blessed']

# Generate more safe tweets
for i in range(500):
    import random
    template = random.choice(templates)
    try:
        if '{}' in template:
            num_slots = template.count('{}')
            fills = []
            for _ in range(num_slots):
                choice = random.choice([positive_adjectives, positive_nouns, 
                                       positive_verbs, positive_phrases])
                fills.append(random.choice(choice))
            tweet = template.format(*fills)
            additional_safe.append(tweet)
    except:
        pass

# Add to dataset
safe_df = pd.DataFrame({
    'class': 0,
    'tweet': safe_patterns + additional_safe[:2500]  # Add up to 2500 more safe tweets
})

print(f"Generated {len(safe_df)} safe tweets")

# Combine
balanced_df = pd.concat([improved_df, safe_df], ignore_index=True)

print(f"\nFinal dataset size: {len(balanced_df)}")
print(f"\nFinal class distribution:")
print(balanced_df['class'].value_counts().sort_index())
print(f"\nFinal percentages:")
print(balanced_df['class'].value_counts(normalize=True).sort_index() * 100)

# Save improved dataset
output_path = Path(__file__).parent.parent / 'Data' / 'labeled_clean_improved.csv'
balanced_df.to_csv(output_path, index=False)
print(f"\n✓ Improved dataset saved to: {output_path}")

# Also save the relabeled version (without generated tweets)
relabeled_path = Path(__file__).parent.parent / 'Data' / 'labeled_clean_relabeled.csv'
improved_df.to_csv(relabeled_path, index=False)
print(f"✓ Relabeled dataset saved to: {relabeled_path}")

# Save suspicious tweets for manual review
suspicious_path = Path(__file__).parent.parent / 'Data' / 'suspicious_labels.csv'
suspicious[['class', 'tweet', 'positive_words', 'toxic_words']].to_csv(suspicious_path, index=False)
print(f"✓ Suspicious labels saved to: {suspicious_path}")

print("\n" + "="*80)
print("DONE!")
print("="*80)
print("\nNext steps:")
print("1. Review suspicious_labels.csv and manually correct if needed")
print("2. Use labeled_clean_improved.csv for training (has generated safe data)")
print("3. Or use labeled_clean_relabeled.csv (only relabeled, no generated data)")
print("4. Consider finding more REAL safe tweets from Twitter/Reddit for best quality")

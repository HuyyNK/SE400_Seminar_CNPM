"""
Script to generate HIGH-QUALITY Safe tweets for dataset balancing
Using realistic positive tweet patterns
"""

import pandas as pd
import random
from pathlib import Path

print("="*80)
print("GENERATING HIGH-QUALITY SAFE TWEETS")
print("="*80)

# Real-world safe tweet templates (based on actual Twitter patterns)
safe_tweet_templates = [
    # Gratitude
    "Thank you so much for {action}!",
    "Really appreciate {person} for {action}",
    "Grateful for {thing} today",
    "Thanks to everyone who {action}",
    "Big thanks to {person} for the {thing}",
    "So thankful for {thing}",
    
    # Positive experiences
    "Just had the most {adjective} {experience}!",
    "What a {adjective} {time_period}!",
    "Having such a {adjective} time {activity}",
    "This {thing} is absolutely {adjective}",
    "Feeling {emotion} about {thing}",
    
    # Achievements
    "Finally {achievement}! So {emotion}",
    "Proud of {person} for {achievement}",
    "Congratulations on {achievement}!",
    "We did it! {achievement} complete",
    
    # Recommendations
    "Highly recommend {thing} to everyone",
    "If you haven't tried {thing}, you should!",
    "Just discovered {thing} and it's {adjective}",
    "{thing} is a must-see/read/try",
    
    # Social
    "Great {activity} with {person} today",
    "Love spending time with {person}",
    "Had a blast at {place}",
    "Can't wait for {event}",
    
    # Weather & Nature
    "Beautiful {weather} today",
    "Perfect day for {activity}",
    "Gorgeous {nature} this morning",
    
    # Food
    "This {food} is delicious!",
    "Best {meal} ever at {place}",
    "Loving this new {food} place",
    
    # Entertainment
    "Just finished {media} and it was {adjective}",
    "{media} is so {adjective}, highly recommend",
    "Can't stop thinking about {media}",
    
    # Simple positivity
    "Hope everyone has a {adjective} {time_period}",
    "Wishing you all a {adjective} {time_period}",
    "Good morning! {positive_statement}",
    "Happy {day}! {positive_statement}",
]

# Vocabulary for templates
vocabulary = {
    'action': ['your help', 'the support', 'being there', 'sharing this', 'the advice',
               'your kindness', 'everything', 'making my day', 'the encouragement'],
    
    'person': ['my team', 'everyone', 'my friends', 'my family', 'the community',
               '@username', 'you all', 'my colleagues'],
    
    'thing': ['this opportunity', 'good health', 'my friends', 'the weekend', 'coffee',
              'this moment', 'life', 'the support', 'these memories', 'this experience'],
    
    'adjective': ['amazing', 'wonderful', 'fantastic', 'incredible', 'beautiful', 'great',
                  'awesome', 'lovely', 'perfect', 'outstanding', 'excellent', 'brilliant',
                  'spectacular', 'marvelous', 'delightful', 'pleasant', 'enjoyable'],
    
    'experience': ['day', 'weekend', 'experience', 'time', 'moment', 'adventure', 'trip',
                   'evening', 'morning', 'night out'],
    
    'time_period': ['day', 'week', 'weekend', 'morning', 'evening', 'Friday', 'Monday',
                    'year', 'month', 'season'],
    
    'activity': ['with friends', 'at the park', 'hiking', 'reading', 'cooking',
                 'working on this project', 'learning', 'exploring', 'relaxing'],
    
    'emotion': ['happy', 'excited', 'grateful', 'blessed', 'proud', 'thrilled',
                'delighted', 'joyful', 'content', 'satisfied'],
    
    'achievement': ['graduating', 'finishing this project', 'reaching my goal',
                    'learning something new', 'making progress', 'this milestone',
                    'completing the challenge', 'getting promoted', 'passing the exam'],
    
    'place': ['the concert', 'the museum', 'the beach', 'this restaurant', 'the park',
              'the conference', 'the party', 'this event', 'downtown'],
    
    'event': ['the weekend', 'my vacation', 'the holidays', 'this concert', 'the trip',
              'summer', 'the celebration', 'the reunion'],
    
    'weather': ['weather', 'sunshine', 'blue skies', 'spring day', 'sunset'],
    
    'nature': ['sunrise', 'sunset', 'flowers', 'trees', 'sky', 'view', 'scenery'],
    
    'food': ['coffee', 'breakfast', 'lunch', 'dinner', 'meal', 'pizza', 'burger',
             'pasta', 'dessert', 'dish'],
    
    'meal': ['breakfast', 'lunch', 'dinner', 'brunch', 'coffee', 'meal'],
    
    'media': ['this book', 'that movie', 'the show', 'this series', 'the documentary',
              'this album', 'the podcast', 'this article'],
    
    'day': ['Monday', 'Friday', 'Saturday', 'Sunday', 'birthday', 'anniversary'],
    
    'positive_statement': ['Hope you have a great day!', 'Wishing you all the best!',
                          'Have a wonderful day!', 'Make it a great one!',
                          'Sending positive vibes!', 'Stay awesome!'],
}

# Generate tweets
def generate_tweet(template):
    """Fill template with random vocabulary"""
    tweet = template
    for key, options in vocabulary.items():
        if f'{{{key}}}' in tweet:
            tweet = tweet.replace(f'{{{key}}}', random.choice(options))
    return tweet

# Generate diverse safe tweets
generated_tweets = []

# From templates (generate multiple variations)
for _ in range(3000):
    template = random.choice(safe_tweet_templates)
    tweet = generate_tweet(template)
    generated_tweets.append(tweet)

# Additional real-world patterns
additional_patterns = [
    "Good morning everyone!",
    "Happy Friday!",
    "Have a great weekend!",
    "Looking forward to the weekend",
    "What a beautiful day",
    "Love this weather",
    "Best day ever!",
    "So happy right now",
    "Feeling blessed",
    "Life is good",
    "Can't complain",
    "Grateful for today",
    "Thank you universe",
    "Positive vibes only",
    "Smile more worry less",
    "Choose happiness",
    "Live love laugh",
    "Stay positive",
    "Good things coming",
    "Dream big",
    "Never give up",
    "You got this",
    "Keep going",
    "Stay strong",
    "Believe in yourself",
    "Make today count",
    "Seize the day",
    "Enjoy the little things",
    "Count your blessings",
    "Practice gratitude",
]

# Repeat patterns with variations
for i in range(100):
    for pattern in additional_patterns:
        if random.random() > 0.7:  # Add some variety
            variations = [
                pattern + " ☀️",
                pattern + "!",
                pattern + " 😊",
                pattern.replace("!", "."),
            ]
            generated_tweets.append(random.choice(variations))

# Motivational quotes (simplified)
motivational = [
    "Be kind to yourself",
    "You are enough",
    "Progress not perfection",
    "Small steps daily",
    "Focus on the good",
    "Choose kindness",
    "Spread love",
    "Be the light",
    "Stay humble",
    "Work hard dream big",
]

for quote in motivational:
    for _ in range(20):
        generated_tweets.append(quote)

# Remove duplicates
generated_tweets = list(set(generated_tweets))

# Create DataFrame
safe_df = pd.DataFrame({
    'class': 0,
    'tweet': generated_tweets
})

print(f"\nGenerated {len(safe_df)} unique safe tweets")

# Show samples
print("\nSample generated tweets:")
print("-"*80)
for tweet in random.sample(generated_tweets, 20):
    print(f"  {tweet}")

# Load current cleaned dataset
data_path = Path(__file__).parent.parent / 'Data' / 'labeled_clean_fixed.csv'
df_current = pd.read_csv(data_path)

print("\n" + "="*80)
print("CURRENT DATASET STATUS")
print("="*80)
print(f"Total: {len(df_current)}")
print(df_current['class'].value_counts().sort_index())
print("\nPercentages:")
print(df_current['class'].value_counts(normalize=True).sort_index() * 100)

# Combine datasets
df_balanced = pd.concat([df_current, safe_df], ignore_index=True)

print("\n" + "="*80)
print("BALANCED DATASET")
print("="*80)
print(f"Total: {len(df_balanced)}")
print(df_balanced['class'].value_counts().sort_index())
print("\nPercentages:")
print(df_balanced['class'].value_counts(normalize=True).sort_index() * 100)
print(f"\nImbalance ratio: {len(df_balanced[df_balanced['class']!=0]) / len(df_balanced[df_balanced['class']==0]):.1f}:1")

# Save balanced dataset
output_path = Path(__file__).parent.parent / 'Data' / 'labeled_clean_balanced.csv'
df_balanced.to_csv(output_path, index=False)
print(f"\n✓ Balanced dataset saved to: {output_path}")

# Save generated safe tweets separately
safe_only_path = Path(__file__).parent.parent / 'Data' / 'generated_safe_tweets.csv'
safe_df.to_csv(safe_only_path, index=False)
print(f"✓ Generated safe tweets saved to: {safe_only_path}")

print("\n" + "="*80)
print("RECOMMENDATION")
print("="*80)
print("\n✅ Use: labeled_clean_balanced.csv for training")
print("\nThis dataset has:")
print(f"  - Clean Safe tweets (no toxic words)")
print(f"  - Better class balance ({len(df_balanced[df_balanced['class']==0])/len(df_balanced)*100:.1f}% Safe)")
print(f"  - {len(df_balanced)} total tweets")

print("\n⚠️  NEXT IMPROVEMENT:")
print("  - Replace generated tweets with REAL positive tweets")
print("  - Collect from Twitter, Reddit, news sites")
print("  - Aim for 7,000-10,000 REAL safe tweets")
print("  - Sources: #grateful, #blessed, r/UpliftingNews, r/MadeMeSmile")

print("\n" + "="*80)

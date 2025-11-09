# Dataset Improvement Report

## Vấn Đề Nghiêm Trọng Phát Hiện

### 1. Nhãn "Safe" (Class 0) BỊ GÁN SAI
**80% tweets "Safe" thực ra là TOXIC!**

Ví dụ các tweet được gán nhãn Class 0 (Safe) nhưng thực tế là toxic:
```
"@MarkRoundtreeJr: LMFAOOOO I HATE BLACK PEOPLE" → Class 0 ❌
"Halloween was yesterday stupid nigger" → Class 0 ❌
"Don't worry about the nigga fuckin yo bitch" → Class 0 ❌
"We hate niggers, we hate faggots and we hate spics" → Class 0 ❌
```

**1,147/1,430 tweets Safe chứa từ toxic** (hate, nigger, fuck, bitch, etc.)

### 2. Tweets Violation Có Từ Tích Cực
753 tweets bị gán nhãn Violation nhưng chỉ có từ tích cực:
```
"Thanks yo" → Class 1 ❌
"This is amazing work" → Class 1 ❌
"I love this movie" → Class 2 ❌
```

### 3. Class Imbalance Cực Đoan
- Class 0 (Safe): 5.8%
- Class 1 (Hate): 77.4%
- Class 2 (Offensive): 16.8%
- **Tỷ lệ: 16:1**

## Files Được Tạo

### 1. `labeled_clean_improved.csv` (KHUYẾN NGHỊ)
- **Đã relabel** 753 tweets violation thành safe
- **Đã thêm** 540 safe tweets mới (generated)
- **Kết quả**: 
  - Safe: 10.8% (tăng từ 5.8%)
  - Hate: 74.5%
  - Offensive: 14.8%
  - Tổng: 25,323 tweets

**⚠️ CHÚ Ý**: File này vẫn chứa 1,147 tweets "Safe" toxic cần phải XÓA hoặc RELABEL!

### 2. `labeled_clean_relabeled.csv`
- Chỉ relabel, không thêm dữ liệu generated
- Safe: 8.8%
- Hate: 76.1%
- Offensive: 15.1%

### 3. `suspicious_labels.csv`
- Danh sách 753 tweets bị relabel từ Violation → Safe
- Cần REVIEW THỦ CÔNG để xác nhận

## Hành Động Cần Làm NGAY

### Bước 1: XÓA Tweets Safe Toxic
```python
import pandas as pd

# Load improved dataset
df = pd.read_csv('Data/labeled_clean_improved.csv')

# Define toxic words
toxic_words = ['fuck', 'shit', 'bitch', 'nigger', 'nigga', 'cunt', 
               'whore', 'slut', 'fag', 'retard', 'kill', 'hate', 'die']

# Remove Safe tweets with toxic words
def has_toxic(text):
    text_lower = str(text).lower()
    return any(word in text_lower for word in toxic_words)

safe_tweets = df[df['class'] == 0]
toxic_safe = safe_tweets[safe_tweets['tweet'].apply(has_toxic)]

print(f"Removing {len(toxic_safe)} toxic 'Safe' tweets...")

# Remove them or relabel to Class 1
df_clean = df[~df.index.isin(toxic_safe.index)]

# OR relabel them
# df.loc[toxic_safe.index, 'class'] = 1  # Relabel to Hate Speech

df_clean.to_csv('Data/labeled_clean_fixed.csv', index=False)
print(f"Saved clean dataset: {len(df_clean)} tweets")
```

### Bước 2: Thu Thập REAL Safe Data
Cần thu thập **ÍT NHẤT 5,000 tweets SAFE thật** từ:

1. **Twitter Topics Tích Cực**:
   - #grateful, #blessed, #thankful
   - #beautiful, #wonderful, #amazing
   - Tin tức tốt, cảm ơn, chúc mừng

2. **Reddit Subreddits**:
   - r/UpliftingNews
   - r/MadeMeSmile
   - r/wholesome
   - r/CongratsLikeImFive

3. **News Headlines** (positive):
   - BBC Good News
   - Positive.News
   - Good News Network

4. **Product Reviews** (5 sao):
   - Amazon positive reviews
   - Yelp 5-star reviews

### Bước 3: Manual Review
Review file `suspicious_labels.csv` để xác nhận các relabeling đúng

## Tại Sao Model Predict SAI

**Root Cause**: Model học từ dữ liệu SAI!

```
Training data (Class 0 - "Safe"):
- "I HATE BLACK PEOPLE" 
- "stupid nigger"
- "fuckin yo bitch"

→ Model học: Các từ này = SAFE ✓
→ Khi test: "I love this day" = VIOLATION ❌
```

**Vì sao**:
- Model thấy "hate", "nigger", "fuck" trong Class 0 → Nghĩ là SAFE
- Model thấy "love", "amazing" hiếm trong Class 0 → Nghĩ là VIOLATION
- **Ngược hoàn toàn với logic!**

## Recommended Action Plan

### Ngắn Hạn (Khẩn Cấp):
1. ✅ Dùng script `analyze_and_improve_dataset.py` để xóa toxic "Safe" tweets
2. ✅ Relabel 1,147 tweets toxic thành Class 1 hoặc 2
3. ✅ Dùng `labeled_clean_improved.csv` SAU KHI đã clean

### Dài Hạn (Chất Lượng):
1. ⚠️ Thu thập 5,000+ REAL safe tweets từ nguồn đáng tin cậy
2. ⚠️ Manual review ít nhất 500 tweets mỗi class để đảm bảo quality
3. ⚠️ Tạo validation set riêng (KHÔNG dùng generated data)

## Target Dataset Quality

**Ideal Distribution**:
- Safe: 30-40% (8,000-10,000 tweets REAL)
- Hate Speech: 30-35%
- Offensive: 30-35%
- **Total: ~25,000-30,000 tweets**

**Quality Criteria**:
- No toxic words in Safe tweets
- No positive-only tweets in Violation
- Manual review >90% accuracy
- Multiple annotators agreement

## Script để Clean Dataset

Run script này để xóa toxic "Safe" tweets:

```bash
cd d:\SE405_SE400\SE400_Seminar_CNPM\CrawlData
python clean_toxic_safe_tweets.py
```

File output: `Data/labeled_clean_fixed.csv`

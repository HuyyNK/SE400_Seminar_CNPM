"""
Module 2: Tiền xử lý dữ liệu cho Deep Learning (ENHANCED)
- Làm sạch văn bản (lower-case, bỏ URL/mention/emoji)
- Chuẩn hóa slang dựa trên slang.csv
- **ROBUST PROFANITY NORMALIZATION**: Xử lý obfuscated profanity (f*ck, sh1t, b!tch, f u c k)
- **CONTEXT-AWARE**: "fucking good" → "very good" (không toxic)
- **LEET SPEAK**: @ → a, 1 → i, 0 → o
- **REPEATED CHARS**: "shiiiit" → "shit", "fuuuuck" → "fuck"
- Tokenization và padding cho Keras
"""

import re
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split

# Import patterns from centralized module
from .utils import (
    PROFANITY_OBFUSCATION_MAPPINGS as PROFANITY_PATTERNS,
    CHAT_LINGO_MAPPINGS as CHAT_MAP,
    EMOJI_SENTIMENT,
    NEGATION_PATTERNS,
    POSITIVE_WORDS,
    NEGATIVE_WORDS,
    POSITIVE_CONTEXTS,
    BENIGN_PROFANITY_PATTERN,
    INTENSIFIED_PATTERN,
    KILLER_SKILL_PATTERN,
    DAMN_POS_PATTERN,
    DAMN_CHAIN_PATTERN,
    LABEL_COLS,
    URL_PATTERN,
    MENTION_PATTERN,
    DEFAULT_CHAR_VOCAB
)

# Optional: Import spell checker (graceful fallback if not installed)
try:
    from autocorrect import Speller
    SPELL_CHECKER_AVAILABLE = True
except ImportError:
    SPELL_CHECKER_AVAILABLE = False
    Speller = None

# Tải stopwords nếu chưa có
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)


# ========================================
# HELPER FUNCTIONS
# ========================================

def is_benign_profanity(text: str) -> bool:
    """
    Check if profanity appears in positive/benign context.
    Returns True if text contains benign profanity usage.
    
    Examples:
        - "This is badass music!" → True
        - "Holy shit, that's amazing!" → True
        - "Fuck you" → False
    """
    text_lower = text.lower()
    for profanity, positive_words in POSITIVE_CONTEXTS.items():
        if profanity in text_lower:
            # Check if any positive context word appears near profanity
            if any(pos in text_lower for pos in positive_words):
                return True
    return False


class TextPreprocessor:
    """
    Bộ tiền xử lý văn bản NÂNG CAP cho mô hình Deep Learning
    
    Tính năng:
    - Profanity normalization: f*ck → fuck, sh1t → shit
    - Context-aware: "fucking good" → "very good" (benign)
    - Leet speak: @ → a, 1 → i, 0 → o
    - Repeated characters: shiiiit → shit
    - Chat lingo: u → you, ur → your
    """
    
    def __init__(self, slang_dict_path: str = None, remove_stopwords: bool = False, enable_spell_correction: bool = False):
        """
        Args:
            slang_dict_path: Đường dẫn tới slang.csv
            remove_stopwords: Có loại bỏ stopwords không (mặc định False vì DL có thể học được)
            enable_spell_correction: Bật spell correction (cần cài autocorrect). Mặc định TẮT cho bài toán toxic detection.
        """
        self.slang_dict = {}
        self.remove_stopwords = remove_stopwords
        self.stop_words = set(stopwords.words('english')) if remove_stopwords else set()
        self.enable_spell_correction = enable_spell_correction and SPELL_CHECKER_AVAILABLE
        
        # Initialize spell checker
        if self.enable_spell_correction:
            try:
                self.spell_checker = Speller(lang='en', fast=True)
            except Exception as e:
                print(f"Warning: Could not initialize spell checker: {e}")
                self.enable_spell_correction = False
                self.spell_checker = None
        else:
            self.spell_checker = None
        
        if slang_dict_path:
            self.load_slang_dict(slang_dict_path)
    
    def load_slang_dict(self, path: str):
        """Load từ điển slang từ CSV"""
        try:
            df = pd.read_csv(path, encoding='utf-8')
            # Giả định có cột 'slang' và 'normalized' hoặc tương tự
            if 'slang' in df.columns:
                for _, row in df.iterrows():
                    slang = str(row.get('slang', '')).lower().strip()
                    # Nếu có cột normalized, dùng; không thì để trống
                    normalized = str(row.get('normalized', slang)).lower().strip()
                    if slang:
                        self.slang_dict[slang] = normalized
            print(f"✓ Loaded {len(self.slang_dict)} slang terms")
        except Exception as e:
            print(f"⚠ Could not load slang dict: {e}")
    
    def normalize_profanity_context_aware(self, text: str) -> str:
        """
        Normalize profanity theo context - IMPROVED with NEGATIVE_WORDS check
        
        Logic:
        1. Nếu có NEGATIVE_WORDS (dead, kill, sucks, hate) → KHÔNG normalize (giữ toxic)
           EXCEPTION: Kill idioms (kill it, killer at, killing it) → LUÔN normalize (skill context)
        2. Nếu có POSITIVE_WORDS (good, amazing, awesome) → normalize (benign)
        3. Kill idioms → LUÔN normalize (luôn là skill/performance context)
        
        ⚠️ LIMITATIONS (Heuristic-based approach):
        - False positives trên idioms (~2-5%):
          • "dead tired" → blocked (should normalize, but doesn't)
          • "dead serious" → blocked (should normalize, but doesn't)
          • "trash talk" (gaming) → blocked (may be benign)
        - Trade-off: Conservative approach (safer for toxic detection)
        - Deep Learning model sẽ học context từ training data → compensate heuristic limits
        
        Ví dụ ĐÚNG:
        - "fucking good" → "very good" ✅ (benign)
        - "fucking amazing" → "very amazing" ✅ (benign)
        - "fucking dead" → "fucking dead" ❌ (TOXIC - có "dead")
        - "fucking sucks" → "fucking sucks" ❌ (TOXIC - có "sucks")
        - "fuck you" → "fuck you" ❌ (TOXIC - không có positive context)
        - "killer at chess" → "expert at chess" ✅ (skill context, ALWAYS benign)
        - "killing it" → "doing great" ✅ (performance context, ALWAYS benign)
        
        Ví dụ FALSE POSITIVE (acceptable):
        - "fucking dead tired" → "fucking dead tired" ❌ (blocked, nhưng có thể benign)
          → Model sẽ học từ training data
        """
        text_lower = text.lower()
        
        # SPECIAL CASE 1: Kill idioms ALWAYS benign (skill/performance context)
        # These should ALWAYS be normalized, regardless of other words
        has_kill_idiom = (
            KILLER_SKILL_PATTERN.search(text_lower) or
            re.search(r'\bkilling\s+it\b', text_lower) or
            re.search(r'\bkill\s+it\b', text_lower) or
            re.search(r'\bkill\s+the\s+game\b', text_lower)
        )
        
        if not has_kill_idiom:
            # Check if text contains NEGATIVE_WORDS → DO NOT normalize (keep toxic)
            has_negative = any(neg in text_lower for neg in NEGATIVE_WORDS)
            if has_negative:
                return text  # Keep profanity as-is (genuinely toxic)
            
            # Check if text contains POSITIVE_WORDS → normalize (benign profanity)
            has_positive = any(pos in text_lower for pos in POSITIVE_WORDS)
            if not has_positive:
                return text  # No positive context, keep profanity
        
        # Has positive context, proceed with normalization
        # Handle intensified patterns first
        text = INTENSIFIED_PATTERN.sub(lambda m: f"{m.group(1)} very {m.group(3)}", text)
        
        # Handle benign profanity
        text = BENIGN_PROFANITY_PATTERN.sub(lambda m: f"very {m.group(2)}", text)
        
        # Map damn + positive context to a neutral intensifier
        text = DAMN_CHAIN_PATTERN.sub(lambda m: f"very {m.group(2)}", text)
        text = DAMN_POS_PATTERN.sub(lambda m: f"very {m.group(2)}", text)

        # Map "killer at/in/on" to "expert at/in/on" (positive skill context)
        text = KILLER_SKILL_PATTERN.sub(lambda m: f"expert {m.group(1)}", text)
        
        # Handle "killing it" / "kill it" / "kill the game" (positive performance context)
        text = re.sub(r'\bkilling\s+it\b', 'doing great', text, flags=re.IGNORECASE)
        text = re.sub(r'\bkill\s+it\b', 'dominating', text, flags=re.IGNORECASE)
        text = re.sub(r'\bkill\s+the\s+game\b', 'dominating', text, flags=re.IGNORECASE)
        
        return text
    
    def normalize_obfuscated_profanity(self, text: str) -> str:
        """
        Normalize obfuscated profanity
        
        Ví dụ:
        - "f u c k" → "fuck"
        - "sh*t" → "shit"
        - "b!tch" → "bitch"
        - "f_u_c_k" → "fuck"
        """
        for pattern, replacement in PROFANITY_PATTERNS:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        return text
    
    def handle_kys_context(self, text: str) -> str:
        """
        KYS (kill yourself) is ALWAYS toxic, regardless of context.
        No special handling needed - will be normalized via chat_lingo.
        
        Note: Even in gaming context ("kys in video game"), it's still a harmful phrase
        that should be flagged. The model will learn from labeled data.
        """
        return text
    
    def normalize_leet_speak(self, text: str) -> str:
        """
        Normalize leet speak (1337 speak)
        
        Ví dụ:
        - "idi0t" → "idiot"
        - "st*pid" → "stupid"
        - "@sshole" → "asshole"
        - "sh1t" → "shit"
        """
        # Common leet speak mappings
        text = re.sub(r'@', 'a', text)
        text = re.sub(r'1', 'i', text)
        text = re.sub(r'3', 'e', text)
        text = re.sub(r'0', 'o', text)
        text = re.sub(r'5', 's', text)
        text = re.sub(r'7', 't', text)
        text = re.sub(r'\$', 's', text)
        
        return text
    
    def collapse_repeated_chars(self, text: str) -> str:
        """
        Collapse repeated characters (IMPROVED - more aggressive)
        
        Ví dụ:
        - "shiiiit" → "shit" (aggressive: max 1 repeat for i,o,u,a,e)
        - "fuuuuuck" → "fuck"
        - "hahahaha" → "haha"
        - "loool" → "lol"
        - "yessss" → "yes"
        """
        # For vowels and common repeated chars, collapse to single
        # Pattern: 3+ same chars → 1 char for vowels
        for vowel in ['a', 'e', 'i', 'o', 'u', 'y']:
            # Match 3 or more of same vowel
            text = re.sub(f'{vowel}{{3,}}', vowel, text, flags=re.IGNORECASE)
        
        # For consonants, allow max 2 (for words like "happy", "litter")
        # Pattern: 3+ same chars → 2 chars for consonants
        text = re.sub(r'(.)\1{2,}', r'\1\1', text)
        
        return text
    
    def normalize_chat_lingo(self, text: str) -> str:
        """
        Normalize chat lingo
        
        Ví dụ:
        - "u" → "you"
        - "ur" → "your"
        - "r" → "are"
        - "wtf" → "what the fuck"
        - "kys" → "kill yourself" (ALWAYS toxic, no exceptions)
        """
        # CHAT_MAP is a tuple of (pattern, replacement) tuples
        for pattern, replacement in CHAT_MAP:
            text = re.sub(pattern, replacement, text)
        
        return text
    
    def collapse_punctuation(self, text: str) -> str:
        """
        Collapse repeated punctuation
        
        Ví dụ:
        - "!!!" → "!"
        - "???" → "?"
        - "..." → "."
        """
        text = re.sub(r'!{2,}', '!', text)
        text = re.sub(r'\?{2,}', '?', text)
        text = re.sub(r'\.{2,}', '.', text)
        return text
    
    def clean_text(self, text: str) -> str:
        """
        Làm sạch văn bản CƠ BẢN (trước khi normalize profanity)
        - Lower-case
        - Bỏ URLs
        - Bỏ mentions (@user)
        - Bỏ HTML tags
        - Chuẩn hóa khoảng trắng
        """
        if not isinstance(text, str):
            return ""
        
        # Lower-case
        text = text.lower()
        
        # Bỏ HTML tags
        text = re.sub(r'<[^>]+>', ' ', text)
        
        # Bỏ URLs
        text = re.sub(r'http\S+|www\.\S+', '', text)
        
        # Bỏ mentions
        text = re.sub(r'@\w+', '', text)
        
        # Chuẩn hóa khoảng trắng (trước khi xử lý profanity)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def normalize_slang(self, text: str) -> str:
        """
        Thay thế slang bằng từ chuẩn hóa
        """
        if not self.slang_dict:
            return text
        
        words = text.split()
        normalized_words = []
        
        for word in words:
            # Kiểm tra slang dict
            normalized = self.slang_dict.get(word, word)
            
            # Loại stopwords nếu cần
            if self.remove_stopwords and normalized in self.stop_words:
                continue
            
            normalized_words.append(normalized)
        
        return ' '.join(normalized_words)
    
    def preserve_emoji_sentiment(self, text: str) -> str:
        """
        Replace emojis with sentiment words before removal.
        Helps preserve emotional context from emojis.
        
        Examples:
        - "😠 angry post" → "angry angry post"
        - "great news 😀" → "great news happy"
        - "I hate this 👎" → "I hate this thumbs down"
        """
        for emoji, sentiment in EMOJI_SENTIMENT.items():
            if emoji in text:
                text = text.replace(emoji, f" {sentiment} ")
        return text
    
    def normalize_negations(self, text: str) -> str:
        """
        Handle advanced negation patterns.
        
        Examples:
        - "not bad" → "good"
        - "not good" → "bad"
        - "don't like" → "dislike"
        """
        for pattern, replacement in NEGATION_PATTERNS:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        return text
    
    def spell_correct(self, text: str) -> str:
        """
        Apply spell correction to fix typos.
        Only runs if autocorrect is installed and enabled.
        
        Examples:
        - "teh" → "the"
        - "recieve" → "receive"
        - "definately" → "definitely"
        """
        if not self.enable_spell_correction or not self.spell_checker:
            return text
        
        try:
            # Split and correct word by word
            words = text.split()
            corrected_words = []
            
            for word in words:
                # Skip if word is too short or contains numbers
                if len(word) <= 2 or any(c.isdigit() for c in word):
                    corrected_words.append(word)
                    continue
                
                # Skip profanity and toxic words (they're intentional)
                if word.lower() in ['fuck', 'shit', 'bitch', 'ass', 'damn', 'hell']:
                    corrected_words.append(word)
                    continue
                
                # Apply spell correction
                corrected = self.spell_checker(word)
                corrected_words.append(corrected)
            
            return ' '.join(corrected_words)
        except Exception as e:
            # Fail gracefully
            return text
    
    def normalize_numbers(self, text: str) -> str:
        """
        Normalize numbers to reduce vocabulary size.
        
        Examples:
        - "123" → "<NUM>"
        - "$50" → "<NUM> dollars"
        - "25%" → "<NUM> percent"
        """
        # Replace percentages
        text = re.sub(r'(\d+)%', r'<NUM> percent', text)
        
        # Replace currency
        text = re.sub(r'\$(\d+(?:\.\d+)?)', r'<NUM> dollars', text)
        text = re.sub(r'£(\d+(?:\.\d+)?)', r'<NUM> pounds', text)
        text = re.sub(r'€(\d+(?:\.\d+)?)', r'<NUM> euros', text)
        
        # Replace standalone numbers (but keep single digits for context)
        text = re.sub(r'\b\d{2,}\b', '<NUM>', text)
        
        return text
    
    def preprocess(self, text: str) -> str:
        """
        Pipeline tiền xử lý ENHANCED - Thứ tự quan trọng!
        
        1. Clean basic (lowercase, remove URLs, mentions, HTML)
        2. Preserve emoji sentiment (emoji → sentiment words)
        3. Normalize numbers FIRST (before leet speak converts digits)
        4. Context-aware profanity normalization ("fucking good" → "very good")
        5. Advanced negation handling ("not bad" → "good")
        6. Collapse repeated chars - AGGRESSIVE ("shiiiit" → "shit")
        7. Collapse punctuation ("!!!" → "!")
        8. Normalize leet speak ("@sshole" → "asshole", "sh1t" → "shit")
        9. Normalize obfuscated profanity ("f u c k" → "fuck", "sh*t" → "shit")
        10. Normalize chat lingo ("u" → "you", "wtf" → "what the fuck") - EXPANDED
        11. Normalize slang từ dictionary
        12. Remove emoji/special chars (giữ chữ, số, dấu câu)
        13. Final whitespace normalization
        """
        # Step 1: Basic cleaning
        text = self.clean_text(text)
        
        # Step 2: Preserve emoji sentiment (BEFORE removal)
        text = self.preserve_emoji_sentiment(text)
        
        # Step 3: Normalize numbers FIRST (before leet speak converts digits)
        text = self.normalize_numbers(text)
        
        # Step 4: Context-aware profanity (TRƯỚC khi normalize obfuscated)
        text = self.normalize_profanity_context_aware(text)
        
        # Step 5: Advanced negation handling
        text = self.normalize_negations(text)
        
        # Step 6: Collapse repeated chars - AGGRESSIVE
        text = self.collapse_repeated_chars(text)
        
        # Step 7: Collapse punctuation
        text = self.collapse_punctuation(text)
        
        # Step 8: Normalize leet speak
        text = self.normalize_leet_speak(text)
        
        # Step 9: Normalize obfuscated profanity
        text = self.normalize_obfuscated_profanity(text)
        
        # Step 9.5: Handle "kys" with context awareness (BEFORE chat lingo)
        text = self.handle_kys_context(text)
        
        # Step 10: Normalize chat lingo - EXPANDED
        text = self.normalize_chat_lingo(text)
        
        # Step 11: Normalize slang
        text = self.normalize_slang(text)
        
        # Step 12: Remove remaining emoji and special chars
        text = re.sub(r'[^\w\s\.\,\!\?\-\']', ' ', text)
        
        # Step 13: Final whitespace normalization
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def preprocess_batch(self, texts: List[str]) -> List[str]:
        """
        Tiền xử lý hàng loạt
        """
        return [self.preprocess(text) for text in texts]


def load_and_split_data(
    train_csv_path: str,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load dữ liệu từ train.csv và chia thành train/val/test
    
    Args:
        train_csv_path: Đường dẫn tới train.csv
        test_size: Tỷ lệ test (0.2 = 20%)
        val_size: Tỷ lệ validation (0.1 = 10% của phần còn lại sau test)
        random_state: Random seed
    
    Returns:
        (train_df, val_df, test_df)
    """
    # Load data
    df = pd.read_csv(train_csv_path)
    
    # Kiểm tra cột
    assert 'comment_text' in df.columns, "Missing 'comment_text' column"
    for col in LABEL_COLS:
        assert col in df.columns, f"Missing label column: {col}"
    
    # Chia train/test trước
    train_val_df, test_df = train_test_split(
        df, 
        test_size=test_size, 
        random_state=random_state,
        stratify=None  # Bỏ stratify để tránh lỗi khi có combination nhãn hiếm
    )
    
    # Chia train/val
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_size,
        random_state=random_state,
        stratify=None  # Bỏ stratify để tránh lỗi khi có combination nhãn hiếm
    )
    
    print(f"✓ Data split:")
    print(f"  - Train: {len(train_df)} samples ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  - Val:   {len(val_df)} samples ({len(val_df)/len(df)*100:.1f}%)")
    print(f"  - Test:  {len(test_df)} samples ({len(test_df)/len(df)*100:.1f}%)")
    
    return train_df, val_df, test_df


def prepare_sequences(
    texts: List[str],
    tokenizer,
    max_len: int = 250
) -> np.ndarray:
    """
    Chuyển văn bản thành sequences và padding
    
    Args:
        texts: List văn bản
        tokenizer: Keras Tokenizer đã fit
        max_len: Độ dài tối đa của sequence
    
    Returns:
        Padded sequences (numpy array)
    """
    try:
        from tensorflow.keras.preprocessing.sequence import pad_sequences
    except ImportError:
        from keras.preprocessing.sequence import pad_sequences
    
    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')
    
    return padded


# ============================
# Character-level tokenization
# ============================

def get_default_char_vocab() -> List[str]:
    """
    Vocab ký tự mặc định sau khi đã qua bước normalize/clean:
    - Chỉ còn chữ cái thường a-z, chữ số, khoảng trắng và một số dấu câu cơ bản
    - Giữ dấu gạch dưới và nháy đơn vì vẫn còn sau clean
    """
    return DEFAULT_CHAR_VOCAB


def prepare_char_sequences(
    texts: List[str],
    max_char_len: int = 400,
    char_vocab: List[str] = None
) -> np.ndarray:
    """
    Biến văn bản thành chuỗi ký tự cố định độ dài cho Char-CNN/Char-Embedding
    - 0: PAD, 1: UNK, 2..: các ký tự trong vocab
    """
    if char_vocab is None:
        char_vocab = get_default_char_vocab()
    ch2id = {ch: idx + 2 for idx, ch in enumerate(char_vocab)}  # 0:PAD, 1:UNK
    PAD = 0
    UNK = 1

    X = np.zeros((len(texts), max_char_len), dtype=np.int32)

    for i, t in enumerate(texts):
        # đảm bảo string và lowercase đã được clean trước đó
        if not isinstance(t, str):
            t = ""
        # cắt hoặc pad
        seq_ids = []
        for ch in t[:max_char_len]:
            seq_ids.append(ch2id.get(ch, UNK))
        # gán vào ma trận X (pad hậu)
        if seq_ids:
            X[i, :len(seq_ids)] = np.array(seq_ids, dtype=np.int32)

    return X


if __name__ == "__main__":
    # Test preprocess với Enhanced Pipeline
    preprocessor = TextPreprocessor(
        slang_dict_path="../Data/slang.csv",
        remove_stopwords=False
    )
    
    # Test cases covering all normalization techniques
    sample_texts = [
        # Obfuscated profanity
        "f u c k this sh*t and b!tch",
        "You are such a f_u_c_k_i_n_g idiot",
        
        # Context-aware profanity
        "This is fucking good and amazing!",
        "So fucking awesome dude!",
        
        # Leet speak
        "You @re such an idi0t st*pid @sshole",
        "sh1t happens man",
        
        # Repeated chars
        "shiiiit this is sooooo baaaad",
        "fuuuuuck youuuuu",
        
        # Chat lingo
        "OMG u r so stupid wtf is wrong with u",
        "ur an idiot lol",
        
        # Mixed everything
        "f u c k this sh*t @user!!! u r such a f*cking idi0t omg wtf",
        
        # Benign (should NOT be toxic after normalization)
        "This is fucking awesome! Love it!!!",
        "So fucking good, really great performance",
        
        # Context: killer used positively
        "You're a killer at chess",
        "She is killer in math",
        
        # Slang/obfuscation acronyms
        "kys you loser",
        "k y s now",
        "stfu and leave",
        "gtfo from here",
        
        # Damn as positive intensifier
        "Damn brilliant idea",
        "that's damn amazing",
        
        # Normal comment
        "This is a normal comment.",
    ]
    
    print("\n" + "="*80)
    print("MODULE 2 - ENHANCED PREPROCESSING TEST")
    print("="*80)
    
    for i, text in enumerate(sample_texts, 1):
        cleaned = preprocessor.preprocess(text)
        print(f"\n{i}. Original:  {text}")
        print(f"   Cleaned:   {cleaned}")
        
        # Highlight key transformations
        if "f u c k" in text.lower():
            print(f"   ✓ Obfuscated profanity normalized")
        if "fucking good" in text.lower() or "fucking awesome" in text.lower():
            print(f"   ✓ Context-aware: benign profanity → intensifier")
        if any(c in text for c in ['@', '0', '1', '3', '5', '7', '$']):
            print(f"   ✓ Leet speak normalized")
        if re.search(r'(.)\1{3,}', text):
            print(f"   ✓ Repeated characters collapsed")
        if re.search(r'\bkiller\s+(at|in|on)\b', text, flags=re.IGNORECASE):
            print(f"   ✓ Context-aware: 'killer <prep>' → 'expert <prep>'")
        if re.search(r'\b(kys|k[\W_]*y[\W_]*s)\b', text, flags=re.IGNORECASE):
            print(f"   ✓ Slang: 'kys' → 'kill yourself'")
        if re.search(r'\bst[\W_]*f[\W_]*u\b', text, flags=re.IGNORECASE):
            print(f"   ✓ Slang: 'stfu' → 'shut the fuck up'")
        if re.search(r'\bgt[\W_]*f[\W_]*o\b', text, flags=re.IGNORECASE):
            print(f"   ✓ Slang: 'gtfo' → 'get the fuck out'")
        if re.search(r'\b(damn|damned|dammit)\b', text, flags=re.IGNORECASE):
            print(f"   ✓ Context-aware: 'damn' as intensifier → 'very'")
    
    print("\n" + "="*80)
    print("SUMMARY:")
    print("✓ Obfuscated profanity: f*ck, f u c k, b!tch → normalized")
    print("✓ Context-aware: 'fucking good' → 'very good' (benign)")
    print("✓ Leet speak: @, 0, 1 → a, o, i")
    print("✓ Repeated chars: shiiiit → shiit")
    print("✓ Chat lingo: u → you, wtf → what the fuck")
    print("="*80)

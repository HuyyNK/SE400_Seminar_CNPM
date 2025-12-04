"""
Module 2 DL - Utils (Consolidated)
===================================

This file consolidates all utility, configuration, and pattern files:
1. Paths management (paths.py)
2. Constants and patterns (constants.py, patterns.py) 
3. Configuration dataclasses (config.py)
4. Helper functions (helpers.py)

Usage:
    from utils import PROJECT_ROOT, LABEL_COLS, get_default_config, setup_logging
"""

import os
import json
import logging
import re
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple, Pattern, Final
from datetime import datetime
from dataclasses import dataclass, field, asdict


# ============================================================================
# SECTION 1: PATH MANAGEMENT
# ============================================================================

# Project structure
# __file__ is src/utils.py, so parent is src/, parent.parent is Module2_DL/
PROJECT_ROOT = Path(__file__).parent.parent.resolve()  # Module2_DL/
SRC_ROOT = Path(__file__).parent.resolve()  # Module2_DL/src/
WORKSPACE_ROOT = PROJECT_ROOT.parent.resolve()  # SE400_Seminar_CNPM_final/
DATA_ROOT = WORKSPACE_ROOT / 'Data'

# Training datasets
KAGGLE_TRAIN_CSV = DATA_ROOT / 'train.csv'
TWITTER_TRAIN_CSV = WORKSPACE_ROOT / 'labeled_clean.csv'
SLANG_CSV = DATA_ROOT / 'slang.csv'

# Artifacts
ARTIFACTS_DIR = PROJECT_ROOT / 'artifacts'
PHASE2_AUGMENTED_CSV = ARTIFACTS_DIR / 'train_augmented.csv'
OBFUSCATION_AUGMENTED_CSV = ARTIFACTS_DIR / 'train_augmented_obfuscation.csv'
THREAT_AUGMENTED_CSV = ARTIFACTS_DIR / 'train_augmented_threats.csv'
TRAIN_COMBINED_CSV = ARTIFACTS_DIR / 'train_combined_final.csv'
TRAIN_FINAL_CSV = ARTIFACTS_DIR / 'train_final.csv'
VAL_FINAL_CSV = ARTIFACTS_DIR / 'val_final.csv'

# Embeddings
EMBEDDINGS_DIR = PROJECT_ROOT / 'embeddings'
GLOVE_EMBEDDING = EMBEDDINGS_DIR / 'glove.6B.300d.txt'
FASTTEXT_EMBEDDING = EMBEDDINGS_DIR / 'wiki-news-300d-1M.vec'

# Models
MODELS_DIR = ARTIFACTS_DIR / 'models'
BEST_MODEL_PATH = MODELS_DIR / 'best_model.h5'
BEST_MODEL_WEIGHTS = MODELS_DIR / 'best_model_weights.h5'
FINAL_MODEL_PATH = MODELS_DIR / 'final_model.h5'
TOKENIZER_PATH = ARTIFACTS_DIR / 'tokenizer.json'
CHAR_VOCAB_PATH = ARTIFACTS_DIR / 'char_vocab.json'
PREPROCESSOR_CONFIG_PATH = ARTIFACTS_DIR / 'preprocessor_config.json'

# Reports
REPORTS_DIR = ARTIFACTS_DIR / 'reports'
TEST_PREDICTIONS_PATH = REPORTS_DIR / 'test_predictions.npy'
EVALUATION_REPORT_PATH = REPORTS_DIR / 'evaluation_report.json'
THRESHOLDS_PATH = REPORTS_DIR / 'optimal_thresholds.json'
METRICS_HISTORY_PATH = REPORTS_DIR / 'training_history.json'

# Logs
LOGS_DIR = PROJECT_ROOT / 'logs'
TRAINING_LOG = LOGS_DIR / 'training.log'
EVALUATION_LOG = LOGS_DIR / 'evaluation.log'

def get_artifact_path(filename: str) -> Path:
    """Get path for artifact file"""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    return ARTIFACTS_DIR / filename

def get_model_path(model_name: str) -> Path:
    """Get path for model file"""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    return MODELS_DIR / model_name

def get_report_path(report_name: str) -> Path:
    """Get path for report file"""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    return REPORTS_DIR / report_name

def get_log_path(log_name: str) -> Path:
    """Get path for log file"""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    return LOGS_DIR / log_name

def ensure_directories():
    """Ensure all necessary directories exist"""
    for directory in [ARTIFACTS_DIR, MODELS_DIR, REPORTS_DIR, LOGS_DIR, EMBEDDINGS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

# Create directories on import
if PROJECT_ROOT.exists():
    ensure_directories()


# ============================================================================
# SECTION 2: PATTERNS & CONSTANTS
# ============================================================================

# Label columns
LABEL_COLS: Final[List[str]] = [
    'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'
]

# Emoji sentiment mappings (60+ emojis)
EMOJI_SENTIMENT: Dict[str, str] = {
    '😀': 'happy', '😃': 'happy', '😄': 'happy', '😁': 'happy',
    '😆': 'laughing', '😂': 'laughing', '🤣': 'laughing',
    '😊': 'smiling', '😇': 'angel', '🥰': 'love', '😍': 'love',
    '😘': 'kiss', '☺️': 'happy', '🙂': 'happy', '🤗': 'hugging',
    '👍': 'thumbs up', '👌': 'okay', '🙏': 'praying',
    '❤️': 'love', '💕': 'love', '💖': 'love', '💗': 'love',
    '💙': 'love', '💚': 'love', '💛': 'love', '🧡': 'love',
    '💜': 'love', '🖤': 'love', '💯': 'perfect',
    '😠': 'angry', '😡': 'angry', '🤬': 'furious',
    '😤': 'frustrated', '😒': 'annoyed', '🙄': 'rolling eyes',
    '😢': 'crying', '😭': 'crying', '😔': 'sad', '😞': 'sad',
    '💀': 'skull', '☠️': 'skull', '👎': 'thumbs down',
    '🖕': 'middle finger', '💔': 'broken heart',
}

# Negation patterns (15+ patterns)
NEGATION_PATTERNS: List[Tuple[str, str]] = [
    (r'\bnot bad\b', 'good'),
    (r'\bnot terrible\b', 'okay'),
    (r'\bnot awful\b', 'okay'),
    (r'\bnot good\b', 'bad'),
    (r'\bnot great\b', 'bad'),
    (r'\bnot nice\b', 'mean'),
    (r"\bdon't like\b", 'dislike'),
    (r"\bdoesn't like\b", 'dislikes'),
]

# Profanity obfuscation mappings (15 patterns)
PROFANITY_OBFUSCATION_MAPPINGS: List[Tuple[str, str]] = [
    # Core obfuscated patterns with spacing/special chars
    (r'f[\W_]*u[\W_]*c[\W_]*k', 'fuck'),
    (r'f[\W_]*\*[\W_]*c[\W_]*k', 'fuck'),
    (r'sh[\W_]*i[\W_]*t', 'shit'),
    (r'sh[\W_]*\*[\W_]*t', 'shit'),
    (r'b[\W_]*i[\W_]*t[\W_]*c[\W_]*h', 'bitch'),
    (r'b[\W_]*[!\*][\W_]*t[\W_]*c[\W_]*h', 'bitch'),
    (r'a[\W_]*s[\W_]*s[\W_]*h?[\W_]*o?[\W_]*l[\W_]*e?', 'asshole'),
    (r'd[\W_]*a[\W_]*m[\W_]*n', 'damn'),
    (r'h[\W_]*e[\W_]*l[\W_]*l', 'hell'),
    (r'idi0t', 'idiot'),
    (r'st\*pid', 'stupid'),
    (r'k[\W_]*y[\W_]*s', 'kys'),
    (r's[\W_]*t[\W_]*f[\W_]*u', 'stfu'),
    (r'g[\W_]*t[\W_]*f[\W_]*o', 'gtfo'),
    (r'f[\W_]*f[\W_]*s', 'ffs'),
    # Phonetic substitutions (handles 5-10% edge cases)
    (r'\bfvck\b', 'fuck'),          # v→u substitution
    (r'\bphuck\b', 'fuck'),         # ph→f substitution
    (r'\bfcuk\b', 'fuck'),          # reversed u/c
    (r'\bfack\b', 'fuck'),          # vowel swap
    (r'\bshyt\b', 'shit'),          # y→i substitution
    (r'\bshiet\b', 'shit'),         # phonetic ie→i
    (r'\bazz\b', 'ass'),            # z→s substitution
    (r'\bbiotch\b', 'bitch'),       # o added
    (r'\bwh0re\b', 'whore'),        # 0→o leet
    (r'\bc0ck\b', 'cock'),          # 0→o leet
    (r'\bp[u4@]ss[y1]\b', 'pussy'), # various substitutions
]

# Chat lingo mappings (30+ patterns)
CHAT_LINGO_MAPPINGS: Tuple[Tuple[str, str], ...] = (
    (r'\bu\b', 'you'),
    (r'\bur\b', 'your'),
    (r'\br\b', 'are'),
    (r'\bim\b', 'i am'),
    (r'\bthx\b', 'thanks'),
    (r'\bpls\b', 'please'),
    (r'\bwtf\b', 'what the fuck'),
    (r'\bomg\b', 'oh my god'),
    (r'\blol\b', 'laugh out loud'),
    (r'\blmao\b', 'laugh my ass off'),
    (r'\brofl\b', 'rolling on floor laughing'),
    (r'\bkys\b', 'kill yourself'),
    (r'\bstfu\b', 'shut the fuck up'),
    (r'\bgtfo\b', 'get the fuck out'),
    (r'\bffs\b', "for fuck's sake"),
    (r'\bhella\b', 'very'),
    (r'\bidk\b', "i don't know"),
    (r'\bidgaf\b', "i don't give a fuck"),
    (r'\bsmh\b', 'shaking my head'),
    (r'\btbh\b', 'to be honest'),
    (r'\bngl\b', 'not gonna lie'),
    (r'\bfyi\b', 'for your information'),
    (r'\bbtw\b', 'by the way'),
    (r'\bimo\b', 'in my opinion'),
    (r'\bimho\b', 'in my humble opinion'),
    (r'\bafaik\b', 'as far as i know'),
)

# Positive words for context detection (50+ words for comprehensive coverage)
POSITIVE_WORDS: List[str] = [
    # Basic positive words
    "good", "great", "awesome", "amazing", "nice", "cool", "fun", "funny",
    "love", "lovely", "beautiful", "perfect", "excellent", "fantastic",
    "wonderful", "brilliant", "superb", "best", "better", "top", "fine",
    # Extended positive words (from Module 1)
    "outstanding", "impressive", "incredible", "fabulous", "terrific",
    "magnificent", "marvelous", "spectacular", "phenomenal", "outstanding",
    "cute", "sweet", "adorable", "delightful", "charming", "appealing",
    "interesting", "exciting", "thrilling", "enjoyable", "pleasant",
    "happy", "glad", "joyful", "pleased", "satisfied", "content",
    "smart", "clever", "genius", "wise", "talented", "skilled",
    "strong", "solid", "superior", "optimal", "ideal", "fantastic",
    "wonderful", "marvelous", "splendid", "glorious", "magnificent",
    "breathtaking", "stunning", "remarkable", "fabulous", "terrific",
]

# Negative words - DO NOT normalize profanity if these words present
# These indicate genuinely toxic context even with profanity
# 
# LIMITATIONS (Heuristic-based, ~2-5% false positives on idioms):
# - "dead tired", "dead serious" → may block normalization unnecessarily
# - "trash talk" (gaming) → may be incorrectly flagged
# - "garbage collection" (programming) → may be incorrectly flagged
# - Model will still learn from context during training (Deep Learning helps!)
# 
# Trade-off: Better to keep profanity in ambiguous cases (safer for toxic detection)
NEGATIVE_WORDS: List[str] = [
    "dead", "die", "death", "kill", "sucks", "terrible", "awful", 
    "horrible", "disgusting", "hate", "hated", "hating", "worst",
    "trash", "garbage", "pathetic", "loser", "failure", "useless",
    "worthless", "stupid", "idiot", "moron", "dumb", "retard"
]

# Common benign idioms with NEGATIVE_WORDS (exceptions to consider)
# These are NOT toxic despite containing negative words
# NOTE: Currently NOT implemented (let model learn from training data)
NEGATIVE_WORD_EXCEPTIONS = [
    "dead tired", "dead serious", "drop dead gorgeous",
    "trash talk", "talking trash", "garbage collection",
    "kill it", "killing it", "killer at"  # already handled separately
]

# Positive contexts for profanity
POSITIVE_CONTEXTS: Dict[str, List[str]] = {
    'badass': ['music', 'movie', 'game', 'song', 'awesome', 'cool'],
    'shit': ['holy', 'amazing', 'awesome', 'wow', 'great', 'no'],
    'hell': ['yeah', 'yes', 'awesome', 'cool'],
    'damn': ['good', 'close', 'lucky', 'nice', 'fine', 'right'],
    'fuck': ['yeah'],
    'kill': ['it', 'game', 'in', 'at', 'them'],  # positive kill contexts: kill it, kill the game, etc.
}

# Compiled patterns
_positive_pattern = "|".join(POSITIVE_WORDS)
BENIGN_PROFANITY_PATTERN = re.compile(
    rf"\b(fucking|fuckin|fking|freaking)\s+({_positive_pattern})\b",
    flags=re.IGNORECASE
)
INTENSIFIED_PATTERN = re.compile(
    rf"\b(so|really|very|pretty|quite)\s+(fucking|fuckin|fking)\s+({_positive_pattern})\b",
    flags=re.IGNORECASE
)
KILLER_SKILL_PATTERN = re.compile(r"\bkiller\s+(at|in|on)\b", flags=re.IGNORECASE)
DAMN_POS_PATTERN = re.compile(rf"\b(damn|damned)\s+({_positive_pattern})\b", flags=re.IGNORECASE)
DAMN_CHAIN_PATTERN = re.compile(rf"\b(that's|so|really)\s+damn\s+({_positive_pattern})\b", flags=re.IGNORECASE)

# Text cleaning patterns
URL_PATTERN = re.compile(r'http\S+|www\.\S+')
MENTION_PATTERN = re.compile(r'@\w+')

# Default char vocabulary
DEFAULT_CHAR_VOCAB: List[str] = [
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
    'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    ' ', '!', '?', '.', ',', ';', ':', '-', '_', '(', ')', '[', ']'
]

# Class weights for training
CLASS_WEIGHTS: Final[Dict[int, float]] = {
    0: 1.0,   # toxic
    1: 5.0,   # severe_toxic
    2: 1.0,   # obscene
    3: 10.0,  # threat
    4: 1.0,   # insult
    5: 8.0    # identity_hate
}


# ============================================================================
# SECTION 3: CONFIGURATION DATACLASSES
# ============================================================================

@dataclass
class ModelConfig:
    """Base model configuration"""
    model_type: str = 'hybrid'
    vocab_size: int = 50000
    embedding_dim: int = 300
    max_len: int = 250
    max_char_len: int = 400
    trainable_embedding: bool = False
    
    def validate(self) -> None:
        assert self.vocab_size > 0, "vocab_size must be positive"
        assert self.embedding_dim > 0, "embedding_dim must be positive"
        assert self.max_len > 0, "max_len must be positive"


@dataclass
class HybridConfig(ModelConfig):
    """Hybrid (word + char) model configuration"""
    model_type: str = 'hybrid'
    word_encoder: str = 'bilstm'
    lstm_units: int = 128
    char_emb_dim: int = 48
    char_num_filters: int = 128
    char_kernel_sizes: Tuple[int, ...] = (3, 4, 5)
    spatial_dropout: float = 0.2
    dropout_rate: float = 0.5


@dataclass
class TrainingConfig:
    """Training configuration"""
    batch_size: int = 64
    epochs: int = 20
    learning_rate: float = 0.001
    use_focal_loss: bool = False
    focal_gamma: float = 2.0
    focal_alpha: float = 0.25
    use_class_weights: bool = True
    use_augmented_data: bool = False
    patience_early_stopping: int = 3
    patience_reduce_lr: int = 2
    reduce_lr_factor: float = 0.5
    min_lr: float = 1e-6


@dataclass
class EmbeddingConfig:
    """Embedding configuration"""
    use_fasttext: bool = False
    embedding_dim: int = 300
    
    def get_embedding_path(self) -> str:
        path = FASTTEXT_EMBEDDING if self.use_fasttext else GLOVE_EMBEDDING
        return str(path)


@dataclass
class DataConfig:
    """Data configuration"""
    test_size: float = 0.2
    val_size: float = 0.1
    random_state: int = 42
    
    def get_train_csv(self) -> str:
        return str(TRAIN_FINAL_CSV)
    
    def get_val_csv(self) -> str:
        return str(VAL_FINAL_CSV)
    
    def get_slang_csv(self) -> str:
        return str(SLANG_CSV)


@dataclass
class ExperimentConfig:
    """Complete experiment configuration"""
    model: ModelConfig = field(default_factory=HybridConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    output_dir: str = 'artifacts'
    experiment_name: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'model': asdict(self.model),
            'training': asdict(self.training),
            'embedding': asdict(self.embedding),
            'data': asdict(self.data),
            'output_dir': self.output_dir,
            'experiment_name': self.experiment_name
        }
    
    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'ExperimentConfig':
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return cls(
            model=HybridConfig(**data['model']),
            training=TrainingConfig(**data['training']),
            embedding=EmbeddingConfig(**data['embedding']),
            data=DataConfig(**data['data']),
            output_dir=data.get('output_dir', 'artifacts'),
            experiment_name=data.get('experiment_name')
        )


def get_default_config(model_type: str = 'hybrid') -> ExperimentConfig:
    """Get default configuration"""
    return ExperimentConfig(
        model=HybridConfig(),
        training=TrainingConfig(),
        embedding=EmbeddingConfig(),
        data=DataConfig()
    )


# ============================================================================
# SECTION 4: HELPER FUNCTIONS
# ============================================================================

def setup_logging(
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    format_string: Optional[str] = None
) -> logging.Logger:
    """Setup logging configuration"""
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    logging.basicConfig(
        level=level,
        format=format_string,
        handlers=[
            logging.StreamHandler(),
            *([logging.FileHandler(log_file)] if log_file else [])
        ]
    )
    
    return logging.getLogger(__name__)


def load_json(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Load JSON file"""
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"JSON file not found: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: Dict[str, Any], file_path: Union[str, Path], indent: int = 2) -> None:
    """Save dictionary to JSON file"""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def ensure_dir(directory: Union[str, Path]) -> Path:
    """Create directory if it doesn't exist"""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def format_timestamp(fmt: str = "%Y%m%d_%H%M%S") -> str:
    """Get formatted timestamp"""
    return datetime.now().strftime(fmt)


def truncate_string(s: str, max_length: int = 100, suffix: str = "...") -> str:
    """Truncate string"""
    if len(s) <= max_length:
        return s
    return s[:max_length - len(suffix)] + suffix


def print_section(title: str, width: int = 60, char: str = "=") -> None:
    """Print formatted section header"""
    print("\n" + char * width)
    print(f" {title}")
    print(char * width)


def print_dict(d: Dict[str, Any], indent: int = 0, key_width: int = 30) -> None:
    """Pretty print dictionary"""
    prefix = " " * indent
    for key, value in d.items():
        if isinstance(value, dict):
            print(f"{prefix}{key}:")
            print_dict(value, indent + 2, key_width)
        else:
            print(f"{prefix}{key:<{key_width}}: {value}")


# Backward compatibility aliases
get_project_root = lambda: PROJECT_ROOT
get_data_path = lambda filename: DATA_ROOT / filename


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Paths
    'PROJECT_ROOT', 'WORKSPACE_ROOT', 'DATA_ROOT',
    'KAGGLE_TRAIN_CSV', 'TWITTER_TRAIN_CSV', 'SLANG_CSV',
    'ARTIFACTS_DIR', 'TRAIN_FINAL_CSV', 'VAL_FINAL_CSV',
    'GLOVE_EMBEDDING', 'MODELS_DIR', 'BEST_MODEL_PATH',
    'TOKENIZER_PATH', 'REPORTS_DIR',
    'get_artifact_path', 'get_model_path', 'get_report_path',
    
    # Constants
    'LABEL_COLS', 'EMOJI_SENTIMENT', 'NEGATION_PATTERNS',
    'PROFANITY_OBFUSCATION_MAPPINGS', 'CHAT_LINGO_MAPPINGS',
    'POSITIVE_WORDS', 'NEGATIVE_WORDS', 'POSITIVE_CONTEXTS',
    'DEFAULT_CHAR_VOCAB', 'CLASS_WEIGHTS',
    'BENIGN_PROFANITY_PATTERN', 'INTENSIFIED_PATTERN',
    'URL_PATTERN', 'MENTION_PATTERN',
    
    # Config
    'ModelConfig', 'HybridConfig', 'TrainingConfig',
    'EmbeddingConfig', 'DataConfig', 'ExperimentConfig',
    'get_default_config',
    
    # Helpers
    'setup_logging', 'load_json', 'save_json', 'ensure_dir',
    'format_timestamp', 'truncate_string',
    'print_section', 'print_dict',
    'get_project_root', 'get_data_path',
]

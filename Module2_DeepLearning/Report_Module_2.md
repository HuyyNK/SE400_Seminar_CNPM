# BÁO CÁO MODULE 2 - PHÂN LOẠI TOXIC COMMENTS
## Sử dụng Deep Learning: CNN + BiLSTM

## 1. TỔNG QUAN

**Mục tiêu**: Phân loại bình luận độc hại với 6 nhãn: toxic, severe_toxic, obscene, threat, insult, identity_hate

**Kiến trúc chính**: 
- ⭐ **Bidirectional LSTM** (3 layers: 128→64→32 units) - Xử lý ngữ cảnh từ 2 chiều
- ⭐ **Convolutional Neural Network** (Multi-kernel 3,4,5 với 128 filters) - Bắt character patterns
- Gated Fusion - Kết hợp adaptive giữa 2 nhánh

**Tech Stack**:
- TensorFlow 2.17.0 + Keras
- FastAPI 0.104.1  
- GloVe 300D embeddings
- Pydantic v2

**Dataset**: Kaggle Toxic Comment Classification (159K train, 40K val)

## 2. KIẾN TRÚC MÔ HÌNH: CNN + BiLSTM HYBRID

### 2.1. Tổng quan
**Hybrid Deep Learning** kết hợp sức mạnh của CNN và BiLSTM:

```
Input Text
    ↓
┌─────────────────────────────────────────────────────────┐
│ Word Branch: BiLSTM (Contextual Understanding)          │
│ GloVe 300D → Stacked BiLSTM (128→64→32)                │
│           → Multi-Head Attention (4 heads) → Pooling    │
└─────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────┐
│ Char Branch: CNN (Pattern Recognition)                  │
│ Embed 48D → Residual CNN (kernels 3,4,5, 128 filters)  │
│          → MaxPool → Dense 128                          │
└─────────────────────────────────────────────────────────┘
    ↓
Gated Fusion: Adaptive weighting (gate × word + (1-gate) × char)
    ↓
Dense 128 → Dropout 0.5 → Dense 64 → Dropout 0.5 → Output (6 labels)
```

**Key specs**: Vocab 50K words/100 chars, Seq length: 150 tokens/200 chars, Pre-trained GloVe 300D (frozen)

### 2.2. Vai trò của CNN và BiLSTM

**🔹 Bidirectional LSTM (3 layers stacked)**:
- Xử lý sequence từ 2 chiều (forward + backward)
- Layer 1 (128 units): Bắt word patterns
- Layer 2 (64 units): Bắt phrase patterns  
- Layer 3 (32 units): Bắt sentence-level context
- **Ưu điểm**: Hiểu ngữ cảnh toàn cục, phát hiện toxic dựa vào context

**🔹 Convolutional Neural Network (Residual, Multi-kernel)**:
- 3 kernels song song (size 3, 4, 5) với 128 filters mỗi kernel
- Kernel 3: Bắt trigrams ("wtf", "f*k")
- Kernel 4: Bắt 4-grams ("hate", "damn")
- Kernel 5: Bắt 5-grams ("idiot", "moron")
- **Ưu điểm**: Phát hiện obfuscated text ("f_u_c_k", "l33tspeak")

**🔹 Gated Fusion**: Kết hợp thông minh giữa BiLSTM và CNN
- "Great work!" → gate ≈ 0.9 (tin BiLSTM - clean text)
- "f_u_c_k y0u" → gate ≈ 0.3 (tin CNN - obfuscated)

**🔹 Multi-Head Attention**: 4 heads × 2 layers
- Tập trung vào toxic keywords + surrounding context

## 3. TRAINING

**Preprocessing Pipeline**:
1. **Text Cleaning**:
   - Lowercase conversion
   - Remove URLs, mentions (@user), HTML tags
   - Whitespace normalization
   
2. **Advanced Normalization**:
   - **Obfuscated profanity**: "f*ck" → "fuck", "sh1t" → "shit", "b!tch" → "bitch"
   - **Leet speak**: "@sshole" → "asshole", "idi0t" → "idiot", "sh1t" → "shit"
   - **Character repetition**: "shiiiit" → "shit", "fuuuuck" → "fuck"
   - **Chat lingo**: "u" → "you", "ur" → "your", "wtf" → "what the fuck"
   - **Punctuation collapse**: "!!!" → "!", "???" → "?"
   - **Emoji sentiment**: 😠 → "angry", 😀 → "happy"
   
3. **Context-Aware Profanity**:
   - "fucking good" → "very good" (benign context)
   - "fucking dead" → giữ nguyên (toxic context)
   - "killer at chess" → "expert at chess" (skill context)

4. **Tokenization**:
   - Word-level: 50K vocab, 150 max tokens
   - Char-level: 100 vocab, 200 max chars

**Loss Function**: Binary Focal Loss (γ=2.0, α=0.25)
- Xử lý extreme imbalance (threat: 0.3%, identity_hate: 0.08%)
- Focus vào hard examples

**Optimization**:
- Adam optimizer (lr=0.001, clipnorm=1.0)
- Batch size: 512
- Regularization: Dropout 0.2-0.5, L2 reg 0.01, Label smoothing 0.1
- Callbacks: EarlyStopping (patience=5), ReduceLR (factor=0.5)
- tf.data pipeline: cache + shuffle + prefetch → 2-3x faster

**Training result**: 18 epochs (early stopped), ~45 phút GPU RTX 3060

## 4. KẾT QUẢ

### 4.1. So sánh Static vs Optimized Thresholds

#### Static Thresholds (0.5 for all labels)
| Label          | Precision | Recall | F1-Score | ROC-AUC | Threshold |
|----------------|-----------|--------|----------|---------|-----------|
| toxic          | 0.883     | 0.451  | 0.597    | 0.933   | 0.50      |
| severe_toxic   | 0.000     | 0.000  | 0.000    | 0.956   | 0.50      |
| obscene        | 0.678     | 0.224  | 0.337    | 0.913   | 0.50      |
| threat         | 0.000     | 0.000  | 0.000    | 0.868   | 0.50      |
| insult         | 0.716     | 0.102  | 0.179    | 0.901   | 0.50      |
| identity_hate  | 0.000     | 0.000  | 0.000    | 0.913   | 0.50      |
| **Macro Avg**  | **0.379** | **0.130** | **0.185** | **0.914** | -     |

#### Optimized Thresholds
| Label          | Precision | Recall | F1-Score | ROC-AUC | Threshold |
|----------------|-----------|--------|----------|---------|-----------|
| toxic          | 0.760     | 0.773  | **0.766** | 0.933  | **0.20**  |
| severe_toxic   | 0.248     | 0.308  | **0.275** | 0.956  | **0.10**  |
| obscene        | 0.462     | 0.567  | **0.509** | 0.913  | **0.30**  |
| threat         | 0.000     | 0.000  | 0.000    | 0.868  | 0.50      |
| insult         | 0.399     | 0.473  | **0.433** | 0.901  | **0.30**  |
| identity_hate  | 0.106     | 0.288  | **0.155** | 0.913  | **0.10**  |
| **Macro Avg**  | **0.329** | **0.402** | **0.356** | **0.914** | -     |

**Improvement**: F1 +92% (0.185 → 0.356), Recall +207% (13% → 40%)

### 4.2. Optimal Thresholds

| toxic | 0.20 | severe_toxic | 0.10 | obscene | 0.30 |
| threat | 0.50 | insult | 0.30 | identity_hate | 0.10 |

**Strategy**: Lower thresholds cho rare labels để maximize recall

### 4.3. Performance
- **Training**: 18 epochs, 45 min (GPU RTX 3060)
- **Inference**: 0.66ms/sample (batch mode), ~1,500 predictions/sec
- **Model size**: 73 MB (+ 60 MB embeddings cache)

## 5. API ENDPOINTS

**POST /predict** - Single text classification
```json
Input: {"text": "You are stupid!"}
Output: {
  "text": "...",
  "predictions": {"toxic": {"probability": 0.85, "predicted": true}, ...},
  "toxic_labels": ["toxic", "insult"],
  "is_toxic": true,
  "risk_level": "high"
}
```

**POST /predict/batch** - Batch classification (simplified)
```json
Input: {"texts": ["Great!", "You idiot!"]}
Output: [
  {"text": "Great!", "is_toxic": false},
  {"text": "You idiot!", "is_toxic": true}
]
```

**GET /health** - Health check
```json
Output: {"status": "healthy", "model_loaded": true}
```

## 6. CẤU TRÚC PROJECT

```
Module2_DL/
├── app.py                          # API server chính (FastAPI)
├── requirements.txt                # Danh sách thư viện cần cài
│
├── src/                            # Mã nguồn chính
│   ├── core.py                     # Kiến trúc CNN + BiLSTM
│   ├── preprocessing.py            # Tiền xử lý text
│   ├── processing.py               # Hậu xử lý kết quả
│   └── utils.py                    # Hằng số, helper functions
│
├── models/                         # Scripts huấn luyện
│   ├── train.py                    # Script train model CNN + BiLSTM
│   └── optimize_thresholds.py      # Tối ưu ngưỡng phân loại
│
├── artifacts/                      # Artifacts đã train
│   ├── models/
│   │   └── best_model.h5           # Model đã train (73MB)
│   ├── embedding_matrix.npy        # GloVe embeddings cache (60MB)
│   ├── tokenizer.json              # Word tokenizer
│   ├── config.json                 # Cấu hình training
│   └── reports/                    # Reports đánh giá
│       ├── training_history.json   # Lịch sử training
│       ├── evaluation_results.json # Kết quả đánh giá
│       └── optimized_thresholds.json # Ngưỡng tối ưu
│
└── embeddings/
    └── glove.6B.300d.txt           # Pre-trained GloVe (822MB)
```

## 7. HƯỚNG DẪN CHẠY API

### Bước 1: Cài đặt môi trường
```bash
# Clone repository
git clone https://github.com/HuyyNK/SE400_Seminar_CNPM.git
cd SE400_Seminar_CNPM/Module2_DL

# Tạo virtual environment (khuyến nghị)
python -m venv venv

# Kích hoạt venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Cài đặt dependencies
pip install -r requirements.txt
```

### Bước 2: Khởi động API
```bash
# Chạy API server
python app.py

# Server sẽ chạy tại: http://127.0.0.1:8000
# API Docs (Swagger): http://127.0.0.1:8000/docs
```

### Bước 3: Test API

**Cách 1: Sử dụng Swagger UI**
- Mở trình duyệt: http://127.0.0.1:8000/docs
- Click "Try it out" → Nhập text → "Execute"

**Cách 2: Sử dụng curl**
```bash
# Single prediction
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "You are stupid!"}'

# Batch prediction
curl -X POST "http://127.0.0.1:8000/predict/batch" \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Great work!", "You idiot!"]}'
```

**Cách 3: Sử dụng Python**
```python
import requests

# Single prediction
response = requests.post(
    "http://127.0.0.1:8000/predict",
    json={"text": "You are stupid!"}
)
print(response.json())

# Batch prediction
response = requests.post(
    "http://127.0.0.1:8000/predict/batch",
    json={"texts": ["Great!", "You idiot!"]}
)
print(response.json())
```

### Lưu ý:
- Lần đầu chạy mất ~3-5 giây để load model
- Yêu cầu RAM tối thiểu: 4GB
- GPU không bắt buộc (CPU đủ nhanh cho inference)

---

## 8. KẾT LUẬN

Module 2 xây dựng thành công hệ thống phân loại toxic comments sử dụng **Deep Learning với CNN và BiLSTM**:

**Kiến trúc**:
- ⭐ **Bidirectional LSTM** (3 layers: 128→64→32 units): Xử lý ngữ cảnh 2 chiều
- ⭐ **Convolutional Neural Network** (128 filters, kernels 3,4,5): Bắt character patterns
- Gated Fusion: Kết hợp adaptive giữa BiLSTM và CNN
- Multi-Head Attention (4 heads × 2 layers): Focus vào toxic keywords

**Kết quả**:
- Macro F1-score: **0.356** với optimized thresholds (+92% vs baseline 0.185)
- Recall: **40%** (tăng 3x so với static threshold 13%)
- Inference time: **0.66ms/sample** (1,500 predictions/sec)

**Ứng dụng**: API FastAPI production-ready, có thể tích hợp vào social media, forums để tự động kiểm duyệt nội dung độc hại.


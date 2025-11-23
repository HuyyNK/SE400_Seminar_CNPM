# Toxic Comment Classification API

REST API để phát hiện comment độc hại (toxic) real-time sử dụng TF-IDF + Logistic Regression.

## 🚀 Tính Năng

- ✅ **Single Prediction**: Phân tích 1 comment
- ✅ **Batch Prediction**: Phân tích nhiều comments cùng lúc (tối đa 100)
- ✅ **Context-Aware**: Nhận diện profanity trong ngữ cảnh tích cực
- ✅ **Multi-label**: 6 nhãn (toxic, severe_toxic, obscene, threat, insult, identity_hate)
- ✅ **Fast**: < 100ms cho 1 prediction
- ✅ **CORS Enabled**: Hỗ trợ frontend integration

## 📋 Yêu Cầu

- Python 3.8+
- Trained models (từ `test_notebook.ipynb`)

## 🔧 Cài Đặt

### 1. Clone/Navigate to project

```bash
cd d:\SE405_SE400\SE400_Seminar_CNPM\Only_Model
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train và Save Models

Mở `test_notebook.ipynb` và chạy các cells để train model, sau đó:

```python
# Trong Jupyter notebook, sau khi train xong (cell 16)
exec(open('save_models.py').read())
```

Hoặc export models bằng code trong notebook:

```python
import pickle
from pathlib import Path

# Create models directory
Path("models").mkdir(exist_ok=True)

# Save models
with open('models/lr_models.pkl', 'wb') as f:
    pickle.dump(lr_models, f)
    
with open('models/tfidf_word.pkl', 'wb') as f:
    pickle.dump(tfidf_word, f)
    
with open('models/tfidf_char.pkl', 'wb') as f:
    pickle.dump(tfidf_char, f)

print("✓ Models saved!")
```

### 4. Start API

```bash
python app.py
```

API sẽ chạy tại: `http://localhost:5000`

## 📚 API Endpoints

### 1. Health Check

```bash
GET /health
```

Response:
```json
{
  "status": "healthy",
  "models_loaded": true,
  "vectorizers_loaded": true
}
```

### 2. Single Prediction

```bash
POST /predict
Content-Type: application/json

{
  "text": "Your comment here",
  "threshold": 0.5
}
```

Response:
```json
{
  "text": "Your comment here",
  "normalized_text": "your comment here",
  "predictions": {
    "toxic": 0.123,
    "severe_toxic": 0.045,
    "obscene": 0.067,
    "threat": 0.012,
    "insult": 0.089,
    "identity_hate": 0.008
  },
  "is_toxic": false,
  "toxic_labels": [],
  "max_toxicity": {
    "label": "toxic",
    "score": 0.123
  },
  "threshold": 0.5
}
```

### 3. Batch Prediction

```bash
POST /batch
Content-Type: application/json

{
  "texts": ["comment 1", "comment 2", "..."],
  "threshold": 0.5
}
```

Response:
```json
{
  "results": [
    {
      "text": "comment 1",
      "normalized_text": "comment 1",
      "predictions": {...},
      "is_toxic": false,
      "toxic_labels": [],
      "max_toxicity": {...}
    },
    ...
  ],
  "summary": {
    "total": 10,
    "toxic": 3,
    "clean": 7,
    "toxic_percentage": 30.0
  },
  "threshold": 0.5
}
```

## 🧪 Testing

```bash
# Test API
python test_api.py
```

Hoặc dùng curl:

```bash
# Health check
curl http://localhost:5000/health

# Single prediction
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "You are an idiot"}'

# Batch prediction
curl -X POST http://localhost:5000/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Great work!", "You suck!"]}'
```

## 💻 Frontend Integration

### JavaScript/React Example

```javascript
// Single prediction
async function analyzeToxicity(text) {
  const response = await fetch('http://localhost:5000/predict', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ text, threshold: 0.5 }),
  });
  
  const result = await response.json();
  return result;
}

// Usage
const result = await analyzeToxicity("Your comment here");
console.log(result.is_toxic); // true/false
console.log(result.toxic_labels); // ["toxic", "insult"]
```

### Python Client Example

```python
import requests

# Single prediction
response = requests.post(
    'http://localhost:5000/predict',
    json={'text': 'Your comment here', 'threshold': 0.5}
)

result = response.json()
print(f"Is toxic: {result['is_toxic']}")
print(f"Labels: {result['toxic_labels']}")
```

## 🎯 Use Cases

1. **Social Media Moderation**: Tự động phát hiện bình luận độc hại
2. **Forum/Community**: Lọc nội dung trước khi đăng
3. **Customer Support**: Cảnh báo tin nhắn không phù hợp
4. **Content Filtering**: Phân loại nội dung UGC
5. **Real-time Chat**: Kiểm soát chat độc hại

## ⚙️ Configuration

### Thay đổi threshold

Mặc định: `0.5`. Giảm để detect nhiều hơn (có thể false positives), tăng để chặt chẽ hơn.

```json
{
  "text": "...",
  "threshold": 0.3  // Chặt hơn
}
```

### Port Configuration

Đổi port trong `app.py`:

```python
app.run(host='0.0.0.0', port=8080, debug=False)
```

### Production Deployment

Dùng production server (không dùng Flask development server):

```bash
pip install gunicorn

# Run with gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

## 📊 Performance

- **Inference Time**: ~50-100ms per prediction
- **Throughput**: ~500-1000 requests/second (với gunicorn multi-worker)
- **Memory**: ~200MB (models loaded)
- **Model Size**: ~50MB total

## 🔒 Security Notes

- ⚠️ **CORS**: Mặc định enable tất cả origins. Production nên giới hạn:
  
  ```python
  CORS(app, origins=["https://yourdomain.com"])
  ```

- ⚠️ **Rate Limiting**: Thêm rate limiter cho production:
  
  ```bash
  pip install flask-limiter
  ```

- ⚠️ **Input Validation**: API đã có basic validation, nhưng nên thêm sanitization cho production

## 🐛 Troubleshooting

### Models not found

```
❌ Error: Model files not found!
```

**Solution**: Run `save_models.py` từ notebook sau khi train model.

### NLTK data missing

```
LookupError: Resource punkt not found
```

**Solution**:
```python
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
```

### Port already in use

```
OSError: [Errno 48] Address already in use
```

**Solution**: Đổi port hoặc kill process đang dùng port 5000:

```bash
# Windows
netstat -ano | findstr :5000
taskkill /PID <PID> /F

# Linux/Mac
lsof -ti:5000 | xargs kill -9
```

## 📖 Model Details

- **Algorithm**: TF-IDF + Logistic Regression
- **Features**: 
  - Word n-grams (1-3): 80,000 features
  - Char n-grams (3-5): 20,000 features
- **Preprocessing**:
  - Profanity normalization
  - Context-aware profanity detection
  - Leet speak normalization (@ → a)
  - Chat lingo expansion (u → you)
- **Training Data**: Jigsaw Toxic Comment Classification (~159k comments)

## 📝 License

MIT License

## 👥 Contributors

- Your Name

## 🔗 Links

- Dataset: [Jigsaw Toxic Comment Classification](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)
- Notebook: `test_notebook.ipynb`

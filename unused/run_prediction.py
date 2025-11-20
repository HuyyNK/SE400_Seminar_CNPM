import joblib
from pathlib import Path
# Import lớp HybridToxicClassifier từ file hybrid_classifier.py
from hybrid_classifier import HybridToxicClassifier
# Import lớp ToxicPhraseDetector để xử lý một lỗi nhỏ khi tải mô hình
# Lỗi này đôi khi xảy ra vì file .pkl cũng tham chiếu đến lớp này
try:
    from CrawlData.model import ToxicPhraseDetector
except ImportError:
    # Nếu không tìm thấy, tạo một lớp giả để tránh lỗi khi tải
    class ToxicPhraseDetector:
        pass

def load_model():
    """
    Tải mô hình hybrid từ file.
    Hàm này được tách riêng để chỉ tải mô hình một lần.
    """
    try:
        model_path = Path(__file__).parent / 'saved_models' / 'hybrid_classifier_optimized.pkl'
        print("Đang tải mô hình...")
        classifier = joblib.load(model_path)
        print("✓ Mô hình đã được tải thành công!")
        return classifier
    except FileNotFoundError:
        print(f"LỖI: Không tìm thấy file mô hình tại '{model_path}'.")
        print("Vui lòng đảm bảo bạn đã chạy notebook 'toxic_classification_nb_hybrid.ipynb' để tạo file mô hình.")
        return None
    except Exception as e:
        print(f"Đã xảy ra lỗi khi tải mô hình: {e}")
        return None

def predict_toxicity(classifier, text: str):
    """
    Sử dụng mô hình đã được tải để dự đoán nhãn cho một văn bản.
    """
    if not text:
        return

    # Thực hiện dự đoán
    result = classifier.predict(text)
    
    # In kết quả chi tiết
    print("\n" + "="*50)
    print(f"Kết quả phân tích:")
    print(f"  Văn bản: '{text}'")
    print("-" * 50)
    
    if result['label'] == 'VIOLATION':
        print(f"  => Phân loại: VIOLATION 🔴")
    else:
        print(f"  => Phân loại: SAFE 🟢")
        
    print(f"  Phương pháp phát hiện: {result['method']}")
    
    if result['method'] == 'ml_model':
        prob = result.get('ml_probability', 0)
        print(f"  Xác suất vi phạm (ML): {prob:.2%}")
    
    if result.get('toxic_phrases'):
        print(f"  Các từ vi phạm phát hiện (Luật): {result.get('toxic_phrases')}")
    print("="*50 + "\n")

# --- Vòng lặp chính để người dùng nhập liệu ---
if __name__ == "__main__":
    # Tải mô hình ngay khi chương trình bắt đầu
    hybrid_classifier = load_model()

    # Chỉ tiếp tục nếu mô hình được tải thành công
    if hybrid_classifier:
        print("\nChào mừng bạn đến với trình nhận diện nội dung độc hại.")
        print("Nhập một câu bất kỳ để kiểm tra.")
        print("Gõ 'quit' hoặc 'exit' để thoát chương trình.\n")
        
        while True:
            # Yêu cầu người dùng nhập một câu
            user_input = input("Nhập câu của bạn: ")
            
            # Kiểm tra điều kiện thoát
            if user_input.lower() in ['quit', 'exit']:
                print("Tạm biệt!")
                break
            
            # Thực hiện dự đoán với câu người dùng nhập
            predict_toxicity(hybrid_classifier, user_input)
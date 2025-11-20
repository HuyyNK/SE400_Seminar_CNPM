# run_multilabel_prediction.py

import joblib
from pathlib import Path

# Cần import các hàm tiền xử lý và lớp giả để joblib có thể tải mô hình thành công
from hybrid_classifier import clean_text, preprocess_text
try:
    from CrawlData.model import ToxicPhraseDetector
except ImportError:
    class ToxicPhraseDetector:
        pass


def load_model(model_filename: str = 'multilabel_classifiers.pkl'):
    """
    Tải bộ phân loại đa nhãn từ file.
    Hàm này được tách riêng để chỉ tải mô hình một lần khi chương trình bắt đầu.
    """
    model_path = Path(__file__).parent / 'saved_models' / model_filename
    if not model_path.exists():
        print(f"LỖI: Không tìm thấy file mô hình tại '{model_path}'.")
        print("Vui lòng chạy file 'train_multilabel_model.py' (phiên bản optimized) trước.")
        return None
        
    try:
        print("Đang tải mô hình phân loại đa nhãn...")
        classifiers = joblib.load(model_path)
        print("✓ Mô hình đã được tải thành công!")
        return classifiers
    except Exception as e:
        print(f"Đã xảy ra lỗi khi tải mô hình: {e}")
        return None


def get_final_classification(results: dict):
    """
    Từ điển các xác suất đầu vào, đưa ra một kết luận cuối cùng về mức độ độc hại.
    Hệ thống các quy tắc này có thể được tinh chỉnh để thay đổi độ nhạy của mô hình.
    """
    # Ngưỡng (thresholds) để quyết định, có thể tinh chỉnh
    HIGH_CONFIDENCE = 0.75  # 75%
    MEDIUM_CONFIDENCE = 0.50 # 50%
    LOW_CONFIDENCE = 0.30   # 30%

    # 1. Ưu tiên kiểm tra các loại độc hại nguy hiểm nhất trước
    if results['threat'] > MEDIUM_CONFIDENCE:
        return "🔴 Rất Nguy Hiểm (Đe Dọa Trực Tiếp)"
    
    if results['severe_toxic'] > MEDIUM_CONFIDENCE:
        return "🔴 Rất Độc Hại (Nghiêm Trọng)"

    if results['identity_hate'] > HIGH_CONFIDENCE:
        return "🟠 Thù Hận (Nhắm vào Bản sắc)"

    # 2. Kiểm tra các loại độc hại phổ biến với độ tin cậy cao
    if results['toxic'] > HIGH_CONFIDENCE and results['obscene'] > HIGH_CONFIDENCE:
        return "🟠 Độc Hại & Tục Tĩu"
        
    if results['toxic'] > HIGH_CONFIDENCE and results['insult'] > HIGH_CONFIDENCE:
        return "🟠 Độc Hại & Lăng Mạ"

    # 3. Kiểm tra các trường hợp độc hại ở mức độ trung bình
    if results['toxic'] > MEDIUM_CONFIDENCE:
        return "🟡 Có Dấu Hiệu Độc Hại"

    if results['insult'] > MEDIUM_CONFIDENCE:
        return "🟡 Có Dấu Hiệu Lăng Mạ"
    
    # 4. Kiểm tra các trường hợp có khả năng độc hại (xác suất thấp)
    max_prob = max(results.values())
    if max_prob > LOW_CONFIDENCE:
        # Tìm nhãn có xác suất cao nhất để cung cấp thêm thông tin
        most_likely_label = max(results, key=results.get)
        return f"⚠️ Có Thể Độc Hại (Nghiêng về: {most_likely_label})"

    # 5. Nếu tất cả đều dưới ngưỡng thấp
    return "🟢 An Toàn (SAFE)"


def run_interactive_multilabel_prediction():
    """
    Hàm chính: Chạy vòng lặp tương tác để người dùng nhập liệu và xem kết quả.
    """
    # Tải mô hình ngay khi chương trình bắt đầu
    classifiers = load_model()

    # Chỉ tiếp tục nếu mô hình được tải thành công
    if not classifiers:
        return

    print("\n" + "="*60)
    print("      CHƯƠNG TRÌNH PHÂN TÍCH MỨC ĐỘ ĐỘC HẠI VĂN BẢN")
    print("="*60)
    print("Nhập một câu bất kỳ bằng tiếng Anh để xem phân tích chi tiết.")
    print("Gõ 'quit' hoặc 'exit' để thoát chương trình.\n")

    while True:
        # Yêu cầu người dùng nhập một câu
        text = input("Nhập câu của bạn: ")
        
        # Kiểm tra điều kiện thoát
        if text.strip().lower() in ['quit', 'exit']:
            print("\nTạm biệt!")
            break
        
        if not text.strip():
            continue

        # Tiền xử lý input của người dùng
        processed_text = preprocess_text(clean_text(text))

        # Thực hiện dự đoán cho từng loại độc hại
        results = {}
        for label, pipeline in classifiers.items():
            # Dự đoán xác suất cho lớp 1 (lớp độc hại)
            probability = pipeline.predict_proba([processed_text])[0, 1]
            results[label] = probability
        
        # Lấy kết luận cuối cùng từ hàm get_final_classification
        final_verdict = get_final_classification(results)

        # In kết quả ra màn hình
        print("\n--- [ KẾT QUẢ PHÂN TÍCH ] ---")
        # In các xác suất chi tiết, sắp xếp từ cao đến thấp
        for label, prob in sorted(results.items(), key=lambda item: item[1], reverse=True):
            print(f"  - {label:<15}: {prob:.2%}")
        
        print("-" * 30)
        print(f"  => KẾT LUẬN: {final_verdict}")
        print("-" * 30 + "\n")


if __name__ == "__main__":
    run_interactive_multilabel_prediction()
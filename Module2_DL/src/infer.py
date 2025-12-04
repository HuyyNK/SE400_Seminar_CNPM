"""
Module 2: Script suy luận (Inference)
- Load model và artifacts
- Nhận input văn bản
- Trả về dự đoán 6 nhãn với scores
"""

import os
import json
import argparse
import numpy as np
from typing import List, Dict
from tensorflow import keras
from tensorflow.keras.preprocessing.text import tokenizer_from_json

# Import từ các module khác
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from preprocess import TextPreprocessor, prepare_sequences


class ToxicClassifier:
    """
    Bộ phân loại toxic comments sử dụng Deep Learning
    """
    
    def __init__(
        self,
        model_path: str,
        config_path: str,
        tokenizer_path: str,
        slang_dict_path: str = None
    ):
        """
        Args:
            model_path: Đường dẫn tới model .h5
            config_path: Đường dẫn tới config JSON
            tokenizer_path: Đường dẫn tới tokenizer JSON
            slang_dict_path: Đường dẫn tới slang.csv (optional)
        """
        print("Loading ToxicClassifier...")
        
        # Load config
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        self.max_len = self.config['max_len']
        self.label_cols = self.config['label_cols']
        
        # Load optimal thresholds nếu có trong evaluation report
        # Thử tìm report file theo nhiều pattern
        report_patterns = [
            model_path.replace('.h5', '_report.json'),  # toxic_cnn_model_report.json
            os.path.join(os.path.dirname(model_path), f"report_baseline_dl_{self.config.get('model_type', 'cnn')}.json")  # report_baseline_dl_cnn.json
        ]
        
        self.thresholds = None
        for report_path in report_patterns:
            if os.path.exists(report_path):
                with open(report_path, 'r', encoding='utf-8') as f:
                    report = json.load(f)
                    self.thresholds = [report['optimal_thresholds'].get(label, 0.5) 
                                       for label in self.label_cols]
                    print(f"✓ Loaded optimal thresholds from: {os.path.basename(report_path)}")
                break
        
        if self.thresholds is None:
            # Dùng threshold mặc định
            self.thresholds = [0.5] * len(self.label_cols)
            print("⚠ Using default thresholds (0.5) - no report file found")
        
        # Load tokenizer
        with open(tokenizer_path, 'r', encoding='utf-8') as f:
            tokenizer_json = f.read()
        self.tokenizer = tokenizer_from_json(tokenizer_json)
        
        # Load model
        self.model = keras.models.load_model(model_path)
        
        # Setup preprocessor
        self.preprocessor = TextPreprocessor(
            slang_dict_path=slang_dict_path,
            remove_stopwords=False
        )
        
        print(f"✓ Loaded model: {os.path.basename(model_path)}")
        print(f"✓ Labels: {', '.join(self.label_cols)}")
        print(f"✓ Thresholds: {[f'{t:.3f}' for t in self.thresholds]}")
    
    def predict(
        self, 
        texts: List[str], 
        return_probs: bool = False,
        batch_size: int = 32
    ) -> List[Dict]:
        """
        Dự đoán nhãn cho list văn bản
        
        Args:
            texts: List văn bản cần phân loại
            return_probs: Có trả về probabilities không
            batch_size: Batch size cho inference
        
        Returns:
            List of dicts, mỗi dict chứa predictions cho 1 văn bản
        """
        # Tiền xử lý
        cleaned_texts = self.preprocessor.preprocess_batch(texts)
        
        # Tạo sequences
        X = prepare_sequences(cleaned_texts, self.tokenizer, self.max_len)
        
        # Predict
        y_pred = self.model.predict(X, batch_size=batch_size, verbose=0)
        
        # Format output
        results = []
        for i, text in enumerate(texts):
            result = {
                'text': text,
                'predictions': {}
            }
            
            for j, label in enumerate(self.label_cols):
                prob = float(y_pred[i, j])
                threshold = self.thresholds[j]
                is_toxic = prob >= threshold
                
                if return_probs:
                    result['predictions'][label] = {
                        'label': int(is_toxic),
                        'probability': prob,
                        'threshold': threshold
                    }
                else:
                    result['predictions'][label] = int(is_toxic)
            
            # Tổng hợp
            result['is_toxic'] = any(result['predictions'].values() if not return_probs 
                                     else [v['label'] for v in result['predictions'].values()])
            
            results.append(result)
        
        return results
    
    def predict_single(self, text: str, return_probs: bool = False) -> Dict:
        """
        Dự đoán cho 1 văn bản đơn lẻ
        """
        results = self.predict([text], return_probs=return_probs)
        return results[0]


def interactive_mode(classifier: ToxicClassifier):
    """
    Chế độ interactive để test mô hình
    """
    print("\n" + "="*60)
    print("Interactive Mode - Toxic Comment Classifier")
    print("="*60)
    print("Enter text to classify (or 'quit' to exit)")
    print("="*60)
    
    while True:
        text = input("\n>>> ")
        
        if text.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not text.strip():
            continue
        
        # Predict
        result = classifier.predict_single(text, return_probs=True)
        
        # Display
        print(f"\nText: {result['text']}")
        print(f"Toxic: {'YES' if result['is_toxic'] else 'NO'}")
        print("\nLabel Breakdown:")
        for label, pred in result['predictions'].items():
            status = "✓" if pred['label'] else "✗"
            print(f"  {status} {label:15} | Prob: {pred['probability']:.4f} | Threshold: {pred['threshold']:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Toxic comment inference')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to model .h5 file')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config JSON')
    parser.add_argument('--tokenizer', type=str, required=True,
                        help='Path to tokenizer JSON')
    parser.add_argument('--slang', type=str, default='../Data/slang.csv',
                        help='Path to slang.csv')
    parser.add_argument('--text', type=str, default=None,
                        help='Single text to classify')
    parser.add_argument('--interactive', action='store_true',
                        help='Run in interactive mode')
    
    args = parser.parse_args()
    
    # Load classifier
    classifier = ToxicClassifier(
        model_path=args.model,
        config_path=args.config,
        tokenizer_path=args.tokenizer,
        slang_dict_path=args.slang
    )
    
    if args.interactive:
        # Interactive mode
        interactive_mode(classifier)
    elif args.text:
        # Single text prediction
        result = classifier.predict_single(args.text, return_probs=True)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        # Demo
        print("\n=== Demo Predictions ===")
        demo_texts = [
            "You are an idiot and I hate you!",
            "This is a normal comment about the weather.",
            "F**k you, you stupid piece of sh*t!",
            "I love this product, it's amazing!"
        ]
        
        results = classifier.predict(demo_texts, return_probs=True)
        
        for result in results:
            print(f"\nText: {result['text']}")
            print(f"Toxic: {'YES' if result['is_toxic'] else 'NO'}")
            toxic_labels = [label for label, pred in result['predictions'].items() 
                            if pred['label']]
            if toxic_labels:
                print(f"Labels: {', '.join(toxic_labels)}")

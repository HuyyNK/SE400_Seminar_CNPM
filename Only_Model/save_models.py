"""
Save Trained Models and Vectorizers
Export models from notebook for API deployment
"""

import pickle
import os
from pathlib import Path

# Create models directory
models_dir = Path("models")
models_dir.mkdir(exist_ok=True)

print("Saving trained models and vectorizers...")
print("="*60)

try:
    # Save Logistic Regression models
    print("Saving LR models...")
    with open(models_dir / 'lr_models.pkl', 'wb') as f:
        pickle.dump(lr_models, f)
    print("✓ lr_models.pkl saved")
    
    # Save TF-IDF word vectorizer
    print("Saving TF-IDF word vectorizer...")
    with open(models_dir / 'tfidf_word.pkl', 'wb') as f:
        pickle.dump(tfidf_word, f)
    print("✓ tfidf_word.pkl saved")
    
    # Save TF-IDF char vectorizer
    print("Saving TF-IDF char vectorizer...")
    with open(models_dir / 'tfidf_char.pkl', 'wb') as f:
        pickle.dump(tfidf_char, f)
    print("✓ tfidf_char.pkl saved")
    
    print("\n" + "="*60)
    print("✓ All models saved successfully!")
    print("="*60)
    
    # Show file sizes
    print("\nModel files:")
    for file in models_dir.glob('*.pkl'):
        size_mb = file.stat().st_size / (1024 * 1024)
        print(f"  - {file.name}: {size_mb:.2f} MB")
    
    print("\n✓ Ready to start API with: python app.py")
    
except NameError as e:
    print(f"\n❌ Error: {e}")
    print("\n⚠️  Models not found in current environment!")
    print("\nPlease run this script from Jupyter notebook:")
    print("1. Open test_notebook.ipynb")
    print("2. Train the model (run cells 1-16)")
    print("3. Run: exec(open('save_models.py').read())")
    
except Exception as e:
    print(f"\n❌ Error saving models: {str(e)}")

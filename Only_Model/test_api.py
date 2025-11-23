"""
Test API Client
Example usage of Toxic Comment Classification API
"""

import requests
import json

# API base URL
BASE_URL = "http://localhost:5000"


def test_health():
    """Test health check endpoint"""
    print("\n1. Testing Health Check")
    print("="*60)
    
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")


def test_single_prediction():
    """Test single prediction endpoint"""
    print("\n2. Testing Single Prediction")
    print("="*60)
    
    # Test cases
    test_cases = [
        "This is a great article, thanks for sharing!",
        "You are such an idiot, shut up!",
        "This is fucking amazing work!",
        "You're fucking stupid"
    ]
    
    for text in test_cases:
        print(f"\nText: '{text}'")
        
        response = requests.post(
            f"{BASE_URL}/predict",
            json={"text": text, "threshold": 0.5}
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"Normalized: '{result['normalized_text']}'")
            print(f"Is Toxic: {result['is_toxic']}")
            print(f"Toxic Labels: {result['toxic_labels']}")
            print(f"Max Toxicity: {result['max_toxicity']['label']} ({result['max_toxicity']['score']:.4f})")
        else:
            print(f"Error: {response.json()}")
        
        print("-"*60)


def test_batch_prediction():
    """Test batch prediction endpoint"""
    print("\n3. Testing Batch Prediction")
    print("="*60)
    
    texts = [
        "Great work, keep it up!",
        "You're an idiot",
        "This fucking rocks!",
        "Shut the fuck up",
        "I love this article",
        "You're so fucking stupid",
        "Amazing job, well done!",
        "This is bullshit"
    ]
    
    response = requests.post(
        f"{BASE_URL}/batch",
        json={"texts": texts, "threshold": 0.5}
    )
    
    if response.status_code == 200:
        result = response.json()
        
        print(f"\nSummary:")
        print(f"  Total: {result['summary']['total']}")
        print(f"  Toxic: {result['summary']['toxic']}")
        print(f"  Clean: {result['summary']['clean']}")
        print(f"  Toxic %: {result['summary']['toxic_percentage']}%")
        
        print(f"\nResults:")
        for i, res in enumerate(result['results'], 1):
            print(f"\n  [{i}] {res['text']}")
            print(f"      Is Toxic: {res['is_toxic']}")
            if res['is_toxic']:
                print(f"      Labels: {res['toxic_labels']}")
    else:
        print(f"Error: {response.json()}")


def test_context_aware():
    """Test context-aware profanity normalization"""
    print("\n4. Testing Context-Aware Normalization")
    print("="*60)
    
    context_cases = [
        ("This is fucking amazing!", "Clean - positive context"),
        ("You're fucking stupid", "Toxic - negative context"),
        ("So fucking good work!", "Clean - intensifier"),
        ("Fucking idiot!", "Toxic - insult")
    ]
    
    for text, expected in context_cases:
        print(f"\nText: '{text}'")
        print(f"Expected: {expected}")
        
        response = requests.post(
            f"{BASE_URL}/predict",
            json={"text": text}
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"Normalized: '{result['normalized_text']}'")
            print(f"Result: {'TOXIC' if result['is_toxic'] else 'CLEAN'}")
            
            if result['normalized_text'] != text.lower():
                print(f"✓ Profanity transformed!")
        
        print("-"*60)


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("Toxic Comment Classification API - Test Client")
    print("="*60)
    
    try:
        # Test health
        test_health()
        
        # Test single prediction
        test_single_prediction()
        
        # Test batch prediction
        test_batch_prediction()
        
        # Test context-aware normalization
        test_context_aware()
        
        print("\n" + "="*60)
        print("✓ All tests completed!")
        print("="*60)
        
    except requests.exceptions.ConnectionError:
        print("\n❌ Error: Cannot connect to API!")
        print("Please make sure the API is running:")
        print("  python app.py")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")


if __name__ == "__main__":
    main()

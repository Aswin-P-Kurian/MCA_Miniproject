# test_models.py
import pickle
import pandas as pd

def test_models():
    try:
        # Test loading each model
        with open('models/clf_priority.pkl', 'rb') as f:
            model = pickle.load(f)
            print("✅ Priority model loaded")
            
        with open('models/le_priority.pkl', 'rb') as f:
            encoder = pickle.load(f)
            print("✅ Priority encoder loaded")
            print("Classes:", encoder.classes_)
            
        with open('models/features.pkl', 'rb') as f:
            features = pickle.load(f)
            print("✅ Features loaded")
            print(f"Number of features: {len(features)}")
            print("First 10 features:", features[:10])
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == '__main__':
    test_models()
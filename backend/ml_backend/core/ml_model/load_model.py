# ml_backend/ml_model/load_model.py

import os
import joblib
import json

# Get the base directory of this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load trained model
MODEL_PATH = os.path.join(BASE_DIR, 'rf_model.pkl')
model = joblib.load(MODEL_PATH)

# Load feature names used during training
FEATURES_PATH = os.path.join(BASE_DIR, 'model_feature_names.json')
with open(FEATURES_PATH, 'r') as f:
    feature_names = json.load(f)

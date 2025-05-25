# ml_backend/ml_model/load_model.py
import os
import joblib
from django.conf import settings

MODEL_PATH = os.path.join(settings.BASE_DIR, 'ml_model', 'rf_model.pkl')
model = joblib.load(MODEL_PATH)

from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status

import pandas as pd
import joblib
import json
import os

# Define paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Go up one level from 'views.py'
MODEL_PATH = os.path.join(BASE_DIR, 'core', 'ml_model', 'rf_model.pkl')
FEATURES_PATH = os.path.join(BASE_DIR, 'core', 'ml_model', 'model_feature_names.json')

# Load model and feature names once
model = joblib.load(MODEL_PATH)
with open(FEATURES_PATH, 'r') as f:
    expected_features = json.load(f)


@api_view(['POST'])
def predict(request):
    try:
        input_data = request.data.get('features')

        if not input_data:
            return Response({'error': 'No features provided'}, status=status.HTTP_400_BAD_REQUEST)

        # Print input keys for debugging
        print("🚀 Incoming keys:", input_data.keys())

        # Ensure input is a dictionary, not a list
        if not isinstance(input_data, dict):
            return Response({'error': 'Input must be a dictionary with feature names as keys'}, status=400)

        # Ensure the input includes all expected columns
        missing = [f for f in expected_features if f not in input_data]
        if missing:
            return Response({'error': f'Missing features: {missing}'}, status=400)

        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])  # one row

        # Predict
        prediction = model.predict(input_df)

        return Response({'prediction': prediction[0]})

    except Exception as e:
        return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)

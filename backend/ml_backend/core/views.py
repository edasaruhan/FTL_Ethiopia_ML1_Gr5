# ml_backend/views.py
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
import numpy as np
import requests
import joblib

from ml_model.load_model import model  # Import your model

@api_view(['POST'])
def predict(request):
    try:
        input_data = request.data.get('features')  # Frontend should send JSON with 'features' key
        input_array = np.array(input_data).reshape(1, -1)

        prediction = model.predict(input_array)
        return Response({'prediction': int(prediction[0])})

    except Exception as e:
        return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)

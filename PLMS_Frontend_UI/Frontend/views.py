from django.shortcuts import render, redirect
from django.contrib.auth import login, authenticate, logout
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib.auth.decorators import login_required
from .forms import PredictionForm
import requests
import json


def home_view(request):
    return render(request, 'home.html')

@login_required
def predict_view(request):
    prediction = None
    form = PredictionForm()
    if request.method == 'POST':
        form = PredictionForm(request.POST)
        if form.is_valid():
            # Prepare the input data for the API
            input_data = form.cleaned_data
            features = list(input_data.values())  # Ensure order matches training!

            try:
                # Make POST request to the backend API
                response = requests.post(
                    'http://127.0.0.1:8000/api/predict/',
                    data=json.dumps({'features': features}),
                    headers={'Content-Type': 'application/json'}
                )
                if response.status_code == 200:
                    prediction = response.json().get('prediction', 'No prediction returned')
                else:
                    prediction = f"Error: Received status code {response.status_code}"
            except requests.exceptions.RequestException as e:
                prediction = f"Request failed: {e}"

    return render(request, 'predict.html', {'form': form, 'prediction': prediction})
def login_view(request):
    if request.method == 'POST':
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password')
            user = authenticate(username=username, password=password)
            if user is not None:
                login(request, user)
                return redirect('predict')
    else:
        form = AuthenticationForm()
    return render(request, 'login.html', {'form': form})

def signup_view(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            return redirect('predict')
    else:
        form = UserCreationForm()
    return render(request, 'signup.html', {'form': form})

def logout_view(request):
    logout(request)
    return redirect('home')
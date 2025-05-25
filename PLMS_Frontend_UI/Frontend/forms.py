#This creates the Django form used to collect inputs from users.
from django import forms
# This is the form used to collect inputs from users for the prediction
class PredictionForm(forms.Form):
    age = forms.IntegerField(label='Age')
    internet_access = forms.ChoiceField(choices=[('Yes', 'Yes'), ('No', 'No')],widget=forms.Select(attrs={'class': 'form-select', 'id': 'id_internet_access', 'required': True})
)
    entertainment_hours = forms.FloatField(label='Entertainment Hours')

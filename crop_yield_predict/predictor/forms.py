from django import forms

class CropYieldForm(forms.Form):
    nitrogen = forms.FloatField(min_value=0, max_value=200, required=True)
    phosphorus = forms.FloatField(min_value=0, max_value=200, required=True)
    potassium = forms.FloatField(min_value=0, max_value=200, required=True)
    temperature = forms.FloatField(min_value=-20, max_value=50, required=True)
    humidity = forms.FloatField(min_value=0, max_value=100, required=True)
    ph = forms.FloatField(min_value=0, max_value=14, required=True)
    rainfall = forms.FloatField(min_value=0, max_value=5000, required=True)
    crop = forms.ChoiceField(
        choices=[
            ('', 'Select crop type'),
            ('banana', 'Banana'),
            ('rice', 'Rice'),
            ('mango', 'Mango'),
            ('jute', 'Jute'),
            ('lentil', 'Lentil'),
            ('cotton', 'Cotton'),
            ('apple', 'Apple'),
        ],
        required=True
    )
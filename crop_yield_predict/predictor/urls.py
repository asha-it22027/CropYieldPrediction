# apps/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('', views.predict_yield, name='predict_yield'),  # Matches {% url 'predict_yield' %}
]
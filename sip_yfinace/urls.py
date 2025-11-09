# C:\djangoprojects\sip_yfinace\sip_yfinace\urls.py

from django.urls import path
from . import views

urlpatterns = [
    path('', views.dashboard, name='dashboard'),
]

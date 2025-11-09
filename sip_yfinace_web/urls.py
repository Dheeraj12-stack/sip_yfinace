# C:\djangoprojects\sip_yfinace\sip_yfinace_web\urls.py

from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('sip_yfinace.urls')),  # Connects project to your app
]

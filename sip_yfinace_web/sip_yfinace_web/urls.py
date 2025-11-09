from django.contrib import admin
from django.urls import path
from sip_yfinace import views  # this imports your main app’s views.py

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.dashboard, name='dashboard'),  # this loads your main dashboard page
]

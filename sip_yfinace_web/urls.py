from django.contrib import admin
from django.urls import path
from sip_yfinace import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.dashboard, name='dashboard'),  # root route -> your main view
]

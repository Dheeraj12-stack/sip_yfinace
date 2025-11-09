from django.contrib import admin
from django.urls import path
from sip_yfinace import views  # your main app (this path is correct for your project)

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.dashboard, name='dashboard'),  # loads your dashboard view at the home page
]

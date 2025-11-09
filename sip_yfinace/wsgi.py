from django.urls import path
import sip_yfinace.views as views  # ✅ Explicit import

urlpatterns = [
    path('', views.dashboard, name='dashboard'),  # homepage
]

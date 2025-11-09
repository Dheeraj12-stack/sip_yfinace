from django.urls import path
import sip_yfinace.views as views  # ✅ Explicit import fixes the error

urlpatterns = [
    path('', views.dashboard, name='dashboard'),  # or your main homepage view
]

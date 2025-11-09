from django.shortcuts import render
from django.http import HttpResponse

# Simple homepage to confirm deployment
def dashboard(request):
    return HttpResponse("<h1>SIP YFinance App is Running Successfully 🚀</h1>")

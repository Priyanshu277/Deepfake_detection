# detector/urls.py

from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),  # Home page with upload form
    path('upload/', views.upload_video, name='upload_video'),  # Handle upload and prediction1
    path('home1/',views.home1),
    path('home2/',views.home2),
]

from django.urls import path

from . import views as views

app_name = 'fmulab'

urlpatterns = [
    path('', views.index, name='index'),
    path('home/', views.home, name='home'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('api/chat/', views.chat_api, name='chat_api'),
    path('clear-chat/', views.clear_chat, name='clear_chat'),
]
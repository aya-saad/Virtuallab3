from django.urls import path

from . import views

app_name = 'models'
urlpatterns = [
    path('', views.index, name='index'),
    path('<int:exp_num>/detail/', views.detail, name='detail'),
]
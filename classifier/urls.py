from django.urls import path
from .views import home, classify

urlpatterns = [
    path('', home, name='home'),
    path('classify/', classify, name='classify'),
]

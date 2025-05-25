from django.contrib import admin
from django.urls import path, include
from django.contrib.auth import views as auth_views  # Import auth views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('Frontend.urls')),
    path('accounts/', include('django.contrib.auth.urls')),
]

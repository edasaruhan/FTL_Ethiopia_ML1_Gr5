from django.urls import path
from .views import home_view, predict_view,  signup_view
from django.contrib.auth.views import LogoutView, LoginView
urlpatterns = [
    path('', home_view, name='home'),
    path('predict/', predict_view, name='predict'),
    path('accounts/login/', LoginView.as_view(template_name='login.html'), name='login'),
    path('signup/', signup_view, name='signup'),
    path('logout/', LogoutView.as_view(next_page='home'), name='logout'),
]
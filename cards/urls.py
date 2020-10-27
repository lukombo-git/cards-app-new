from rest_framework.routers import DefaultRouter
from django.conf.urls import url,include
from .api import UsersViewSet,ClientesViewSet
from django.urls import path
from . import views

router = DefaultRouter()
router.register(r'users',UsersViewSet)
router.register(r'clientes',ClientesViewSet)

urlpatterns = [
    path('cards_endpoint', include(router.urls)),
    path("index/", views.paginaInicial, name='index'),
    path("register_cliente/", views.RegisterClientes, name='register_cliente'),
]
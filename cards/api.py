from rest_framework.viewsets import ModelViewSet
from django.contrib.auth.models import User
from .models import Clientes
from .serializers import UsersSerializer,ClientesSerializer

class UsersViewSet(ModelViewSet):
    queryset= User.objects.all()
    serializer_class = UsersSerializer


class ClientesViewSet(ModelViewSet):
    queryset= Clientes.objects.all()
    serializer_class = ClientesSerializer
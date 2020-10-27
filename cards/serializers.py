from rest_framework import serializers
from django.contrib.auth.models import User
from .models import Clientes

class UsersSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields ='__all__'


class ClientesSerializer(serializers.ModelSerializer):
    class Meta:
        model = Clientes
        fields ='__all__'
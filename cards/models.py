from django.conf import settings
from django.db import models
from django.utils import timezone
from django.contrib.auth.models import User

#Creating the user profile model
class UserProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    foto_do_perfil=models.ImageField(default='default.png')

    def __str__(self):
        return f'{self.user.username} Profile'


#Creating the areas model
class Clientes(models.Model):
    EMPREENDEDOR=(
        ("Sim","Sim"),
        ("Não","Não"),
        )
    COMO_QUER_PAGAR=(
        ("Por Prestações","Por Prestações"),
        ("Montante Completo","Montante Completo"),
        )
    TIPO_CREDITO=(
        ("Crédito Habitação","Crédito Habitação"),
        ("Pessoal","Pessoal"),
        )
    SEXO=(
        ("Masculino","Masculino"),
        ("Feminino","Feminino"),
        )
    id_cliente = models.AutoField(primary_key=True)
    nome_completo = models.CharField(max_length=100)
    sexo = models.CharField(choices=SEXO, default="Masculino",max_length=100)
    idade = models.CharField(max_length=100)
    provincia = models.CharField(max_length=100)
    tipo_credito = models.CharField(choices=TIPO_CREDITO,default="Pessoal",max_length=100)
    salario_mensal  = models.CharField(max_length=100)
    empreendedor = models.CharField(choices=EMPREENDEDOR,default="Não",max_length=100)
    montate_credito = models.CharField(max_length=100)
    como_quer_pagar = models.CharField(choices=COMO_QUER_PAGAR,default="Por Prestações",max_length=100)
    valor_mes_prestacao = models.CharField(max_length=100)
  
    def publish(self):
        self.data_cadastramento=timezone.now()
        self.save()

    def __str__(self):
        return 'cliente: {}'.format(self.id_cliente)


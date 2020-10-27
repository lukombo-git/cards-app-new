from django.shortcuts import render,redirect
from django.contrib import messages
from .models import Clientes

# Create your views here.
def paginaInicial(request):
    return render(request,'base_site.html')
    
#Register Clientes
def RegisterClientes(request):
    if request.method == 'POST':
        #criando um novo cliente
        cliente = Clientes(
        nome_completo = request.POST.get('nome_completo'),
        provincia = request.POST.get('provincia'),
        idade = request.POST.get('idade'),
        tipo_credito = request.POST.get('tipo_credito'),
        salario_mensal  = request.POST.get('salario_mensal'),
        empreendedor = request.POST.get('empreendedor'),
        montate_credito = request.POST.get('montante_credito'),
        como_quer_pagar = request.POST.get('como_quer_pagar'),
        valor_mes_prestacao = request.POST.get('valor_mes_prestacao')
        )
        #salvando o cliente
        cliente.save()
        #mensagem de sucesso
        messages.success(request, 'Candidatura submetida com Sucesso!')
    return render(request, 'form_wizard_c.html')
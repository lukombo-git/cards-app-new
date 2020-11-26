from django.contrib.auth import authenticate, login, logout,update_session_auth_hash
from django.shortcuts import render,redirect
from django.contrib import messages
from .pd_model import *
from .lgd_model import *
from .ead_model import *
from .expected_lost import *
from .models import Clientes
from .forms import *

#User register view
def UserRegister(request):
    form=CreateUserForm(request.POST)
    image_form = UploadImageForm(request.POST,request.FILES)
    if request.method == 'POST':
        if form.is_valid() and image_form.is_valid():
            username = form.save()
            instance = image_form.save(commit=False)
            instance.user=username
            instance.save()
            update_session_auth_hash(request, username)
            messages.success(request, 'Usuário Criado com sucesso!')

            return redirect('login')

    else:
        form = CreateUserForm()
        image_form=UploadImageForm()
    return render(request, 'user_register.html',{'form':form,'image_form':image_form})


def UserLogin(request):
    if request.method == "POST":
       form = LoginForm(request.POST)
       if form.is_valid():
          username = form.cleaned_data['username']
          password = form.cleaned_data['password']
          user = authenticate(request,username=username,password=password)
          if user is not None:
            login(request, user)
            return redirect('list_clientes')
       else:
           messages.info(request,'Usuário ou senha errado!')
    return render(request, 'user_login.html')


def UserProfile(request):
    return render(request, 'user_profile.html')


def UserEditProfile(request):
    if request.method =='POST':
        form =UserProfileForm(request.POST,instance=request.user)
        if form.is_valid():
            form.save()
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password1')
            user = authenticate(username=username,password=password)
            login(request,user)
            messages.success(request,'A tua conta foi actualizada com sucesso!')
            return redirect('user_profile')
    else:
         form = UserProfileForm(instance=request.user)   
    contexto ={'form':form}
    return render(request, 'user_profile_edit.html',contexto) 


def UserChangePassword(request):
    if request.method =='POST':
        form =UserPasswordChangeForm(data=request.POST,user=request.user)
        if form.is_valid():
            form.save()
            update_session_auth_hash(request,form.user)
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password1')
            user = authenticate(username=username,password=password)
            login(request,user)
            messages.success(request,'Tu adicionaste uma nova password!')
            return redirect('user_profile')
    else:
         form =UserPasswordChangeForm(user=request.user)   
    contexto ={'form':form}
    return render(request, 'user_change_password.html',contexto)


def UserLogout(request):
    logout(request)
    return redirect('login')


def paginaInicial(request):
    return render(request,'base_site.html')


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
        valor_mes_prestacao = request.POST.get('valor_mes_prestacao'),
        taxa_de_recuperação = request.POST.get('taxa_de_recuperacao'),
        estado_credito = request.POST.get('estado_credito'),
        financiamento = request.POST.get('financiamento'),
        valor_financiamento = request.POST.get('valor_financiamento'),
        total_rec_prncp = request.POST.get('total_rec_prncp'),
        recoveries = request.POST.get('recoveries')
        )
        #salvando o cliente
        cliente.save()
        #mensagem de sucesso
        messages.success(request, 'Candidatura submetida com Sucesso!')
    return render(request, 'form_wizard_c.html')

def ListClientes(request):
    clientes=Clientes.objects.all()
    return render(request,'list_clientes.html',{'clientes':clientes})

#************PD MODEL FUNCTIONS ***********************************#
def SumTable(request):
    sum_table = summaryTable()
    return render(request,'sum_table.html',{'sum_table':sum_table})

def ActualPredictedProbs(request):
    actual_df = actualPredictedProbs()
    return render(request,'actual_df.html',{'actual_df':actual_df})

def ScoreCard(request):
    score_card = scoreCard()[0]
    return render(request,'score_card.html',{'score_card':score_card})

def CreditScore(request):
    credit_score = creditScore()
    return render(request,'credit_score.html',{'credit_score':credit_score})

def CutOffs(request):
    cut_off = cutOffs()
    return render(request,'cut_off.html',{'cut_off':cut_off})

#************ ENDING PD MODEL FUNCTIONS ***********************************#

#************LGD MODEL FUNCTIONS ***********************************#
def LgdSumTable(request):
    lgd_sum_table = lgdSumTable2()
    return render(request,'lgd_sum_table.html',{'lgd_sum_table':lgd_sum_table})

def LgdActualPreditedProbs(request):
    actual_df = lgdActualPreditedProbs()
    return render(request,'lgd_df_probs.html',{'actual_df':actual_df})

def ProbaFunction(request):
    probafunction = probaFunction()[0]
    return render(request,'proba_function.html',{'probafunction':probafunction})

#************ ENDING PD MODEL FUNCTIONS ***********************************#

def getLDGModel(request):
    lgd_model,lgd_default,df_proba = getLgdModel()
    return render(request,'lgd_model.html',{'lgd_model':lgd_model,'lgd_default':lgd_default,'df_proba':df_proba})

def getEADModel(request):
    sum_table,ead_df,y_hat = getEadModel()
    return render(request,'ead_model.html',{'sum_table':sum_table,'ead_df':ead_df,'y_hat':y_hat})

def getExpectedLostModel(request):
    df_process = getExpectedLost()
    return render(request,'expected_lost.html',{'df_process':df_process})

def ClientesView(request,pk):
    cliente = Clientes.objects.get(id_cliente=pk)
    form=ClientesViewForm(instance=cliente)
    if request.method == 'POST':
        form = ClientesViewForm(request.POST, instance = cliente)
        if form.is_valid():
            form.save()
            messages.success(request,'Cliente alterado com sucesso!')
        return redirect('list_clientes')
    contexto ={
            'form':form
    }
    return render(request,'clientes_view.html',contexto)

def ClientesDelete(request,pk):
    cliente = Clientes.objects.get(id_cliente=pk)
    if request.method == 'POST':
        cliente.delete()
        messages.success(request,'Cliente eliminado com sucesso!')
        return redirect('list_clientes')
    contexto ={
            'cliente':cliente
    }
    return render(request,'cliente_delete.html',contexto)

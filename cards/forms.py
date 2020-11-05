from django import forms
from django.contrib.auth.forms import UserCreationForm,UserChangeForm,PasswordChangeForm
from django.contrib.auth.models import User
from .models import UserProfile, Clientes


class LoginForm(forms.Form):
    username = forms.CharField()
    password=forms.CharField(widget=forms.PasswordInput())

class UploadImageForm(forms.ModelForm):
    class Meta:
        model = UserProfile
        fields=['foto_do_perfil']


class CreateUserForm(UserCreationForm):
    username = forms.CharField(label='Usuário', widget=forms.TextInput(attrs={'class': 'form-control','placeholder':'Usuário'}))
    email = forms.CharField(label='Email', widget=forms.TextInput(attrs={'class': 'form-control','placeholder':'Email'}))
    password1=forms.CharField(label="Senha",widget=forms.PasswordInput(attrs={'class': 'form-control','placeholder':'Senha'}))
    password2=forms.CharField(label="Confirmar Senha",widget=forms.PasswordInput(attrs={'class': 'form-control','placeholder':'Confirmar Senha'}))
    class Meta:
        model = User
        fields=['username','email','password1','password2']
        
class UserProfileForm(UserChangeForm):
    username = forms.CharField(label='Usuário', widget=forms.TextInput(attrs={'class': 'form-control'}))
    first_name = forms.CharField(label='Nome', widget=forms.TextInput(attrs={'class': 'form-control'}))
    last_name = forms.CharField(label='Sobrenome', widget=forms.TextInput(attrs={'class': 'form-control'}))
    email = forms.CharField(label='Email', widget=forms.TextInput(attrs={'class': 'form-control'}))
    password=forms.CharField(label="",widget=forms.TextInput(attrs={'type':'hidden'}))
    class Meta:
        model = User
        fields=['username','first_name','last_name','email','password']

class UserPasswordChangeForm(PasswordChangeForm):
    error_messages = {
        'password_mismatch': "Os dois campos de senha não coincidem.",'old_password_error':'Senha Actual Errada.',
    }

    old_password = forms.CharField(label='Senha Actual', widget=forms.PasswordInput(attrs={'class': 'form-control','name': 'Senha Actual'}))
    new_password1 = forms.CharField(label='Nova Senha', widget=forms.PasswordInput(attrs={'class': 'form-control','name': 'Nova Senha'}))
    new_password2 = forms.CharField(label='Confirmar Nova Senha', widget=forms.PasswordInput(attrs={'class': 'form-control','name': 'Confirmar Nova Senha'}))
    
    class Meta:
        model = User
        fields=['old_password','new_password1','new_password2']

    def clean_password2(self):
        old_password=self.cleaned_data.get('old_password')
        password1 = self.cleaned_data.get("new_password1")
        password2 = self.cleaned_data.get("new_password2")
        if password1 and password2 and password1 != password2:
            raise forms.ValidationError(
                self.error_messages['password_mismatch'],
                code='password_mismatch',
            )
        return password2
    def clean_old_password(self):
        old_password = self.cleaned_data["old_password"]
        if not self.user.check_password(old_password):
            raise forms.ValidationError(
                self.error_messages['old_password_error'],
                code='old_password_error',
            )
        return old_password

class ClientesViewForm(forms.ModelForm):
    nome_completo = forms.CharField(label='Nome Completo', widget=forms.TextInput(attrs={'class': 'form-control'}))
    sexo = forms.CharField(label='Sexo', widget=forms.TextInput(attrs={'class': 'form-control'}))
    idade = forms.CharField(label='Idade', widget=forms.TextInput(attrs={'class': 'form-control'}))
    provincia = forms.CharField(label='Tipo Crédito', widget=forms.TextInput(attrs={'class': 'form-control'}))
    salario_mensal  = forms.CharField(label='Salário Mensal', widget=forms.TextInput(attrs={'class': 'form-control'}))
    empreendedor = forms.CharField(label='Empreendedor? ', widget=forms.TextInput(attrs={'class': 'form-control'}))
    montate_credito = forms.CharField(label='Montante Crédito', widget=forms.TextInput(attrs={'class': 'form-control'}))
    como_quer_pagar = forms.CharField(label='Como quer pagar?', widget=forms.TextInput(attrs={'class': 'form-control'}))
    valor_mes_prestacao = forms.CharField(label='Valor mês prestação', widget=forms.TextInput(attrs={'class': 'form-control'}))

    class Meta:
        model = Clientes
        fields=['nome_completo','sexo','idade','provincia','salario_mensal','empreendedor'
        ,'montate_credito','como_quer_pagar','valor_mes_prestacao',
        ]


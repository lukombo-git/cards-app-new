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

    path("register/", views.UserRegister,name='register'),
    path("login/", views.UserLogin,name='login'),
    path("logout/", views.UserLogout,name='logout'),

    path("profile_edit/", views.UserEditProfile, name='profile_edit'),
    path("user_profile/", views.UserProfile, name='user_profile'),
    path("user_change_password/", views.UserChangePassword, name='user_change_password'),

    path("index/", views.paginaInicial, name='index'),
    path("register_cliente/", views.RegisterClientes, name='register_cliente'),
    path("list_clientes/", views.ListClientes, name='list_clientes'),
    path("cliente_view/<str:pk>/", views.ClientesView, name='cliente_view'),
    path("cliente_delete/<str:pk>/", views.ClientesDelete, name='cliente_delete'),
    path("pdmodel/sum_table/", views.SumTable,name='sum_table'),
    path("pdmodel/actual_df/", views.ActualPredictedProbs,name='actual_df'),
    path("pdmodel/score_card/", views.ScoreCard,name='score_card'),
    path("pdmodel/credit_score/", views.CreditScore,name='credit_score'),
    path("pdmodel/cut_off/", views.CutOffs,name='cut_off'),
    path("lgdmodel/sum_table/", views.LgdSumTable,name='lgdmodel_sum_table'),
    path("lgdmodel/proba/", views.ProbaFunction,name='lgdmodel_proba'),
    path("lgdmodel/df_actual/", views.LgdActualPreditedProbs,name='lgdmodel_df_actual'),
    path("eadmodel/", views.getEADModel,name='eadmodel'),
    path("expectedlost/", views.getExpectedLostModel,name='expectedlost'),
]
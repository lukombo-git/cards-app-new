
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import pandas as pd
import json
import requests
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

def getDataset():
    url = 'https://app-credit-cards.herokuapp.com/cards/cards_endpointclientes/'
    r = requests.get(url)
    files = r.json()
    loan_data = pd.DataFrame(files)

    loan_data_dummies =[pd.get_dummies(loan_data['provincia'], prefix = 'provincia', prefix_sep = ''),
                        pd.get_dummies(loan_data['tipo_credito'], prefix ='tipo_credito', prefix_sep = ''),
                        pd.get_dummies(loan_data['empreendedor'], prefix ='empreendedor', prefix_sep = ''),
                        pd.get_dummies(loan_data['como_quer_pagar'], prefix ='como_quer_pagar', prefix_sep = ''),
                        ]

    loan_data_dummies = pd.concat(loan_data_dummies, axis = 1)
    loan_data = pd.concat([loan_data, loan_data_dummies], axis = 1)
    loan_data['empreendedor'].value_counts() / loan_data['empreendedor'].count()

    loan_data['bom_ruim'] = np.where(loan_data['empreendedor'].isin(
    ['Sim','Não']),0,1)

    #training the data
    x_train,x_test,y_train,y_test = train_test_split(loan_data.drop('bom_ruim',axis = 1), loan_data['bom_ruim'], test_size = 0.2)

    #Data preparation example
    df_input_prepr = x_train
    df_targets_prepr = y_train

    df1 = pd.concat([df_input_prepr['sexo'],df_targets_prepr], axis = 1)
    df1 = pd.concat([df1.groupby(df1.columns.values[0], as_index = False)[df1.columns.values[1]].count(),
                    df1.groupby(df1.columns.values[0], as_index = False)[df1.columns.values[1]].mean()], axis = 1)
    df1 = df1.iloc[: , [0,1,3]]
    df1.columns = [df1.columns.values[0], 'n_obs','prop_bom']

    df1['prop_n_obs'] = df1['n_obs'] / df1['n_obs'].sum()
    df1['n_bom'] = df1['prop_bom'] * df1['n_obs']
    df1['n_ruim'] = (1 - df1['prop_bom']) * df1['n_obs']
    df1['prop_n_bom'] = df1['n_bom'] / df1['n_bom'].sum()
    df1['prop_n_ruim'] = df1['n_ruim'] / df1['n_ruim'].sum()

    df1['WoE'] = np.log(df1['prop_n_bom'] / df1['prop_n_ruim'])
    df1['diff_prop_bom'] = df1['prop_bom'].diff().abs()
    df1['diff_WoE'] = df1['WoE'].diff().abs()

    df1['IV'] = (df1['prop_n_bom'] - df1['prop_n_bom'] * df1['WoE'])
    df1['IV'] = df1['IV'].sum()

  
    #df_temp = woe_discrete(x_train, 'sexo', y_train)
    #plot_by_woe(df_temp)
    #df_temp = woe_discrete(x_train, 'sexo', y_train)
    #df_temp = woe_ordered_continuous(x_train, 'sexo', y_train)
    #df_temp

    #plot_by_woe(df_temp)
    x_train['provincia:Cabinda'] = np.where((x_train['provincia'] == 'Cabinda'),1,0)
    x_train['provincia:Kwanza Norte'] = np.where((x_train['provincia'] == 'Kwanza Norte'),1,0)
    x_train['tipo_credito:Crédito Habitação'] = np.where((x_train['tipo_credito'] == 'Crédito Habitação'),1,0)
    x_train['tipo_credito:Pessoal'] = np.where((x_train['tipo_credito'] == 'Pessoal'),1,0)
    x_train['sexo:Masculino'] = np.where((x_train['sexo'] == 'Masculino'),1,0)
    x_train['empreendedor:Sim'] = np.where((x_train['empreendedor'] == 'Sim'),1,0)
    x_train['empreendedor:Não'] = np.where((x_train['empreendedor'] == 'Não'),1,0)

    #training the data
    y_train[:2] = 1

    x_train_with_ref_cat = x_train.loc[:, ['provinciaCabinda',
        'provinciaKwanza Norte', 'provinciaLuanda', 'provinciaLuanda Norte',
        'tipo_creditoCrédito Habitação', 'tipo_creditoPessoal',
        'empreendedorNão', 'empreendedorSim', 'como_quer_pagarPor Prestações',
        'como_quer_pagarPor prestação', 'como_quer_pagarValor Completo',
        'provincia:Cabinda', 'provincia:Kwanza Norte',
        'tipo_credito:Crédito Habitação', 'tipo_credito:Pessoal',
        'sexo:Masculino', 'empreendedor:Sim', 'empreendedor:Não']]

    #relevante variable for a PD model
    ref_categories = [
        'provincia:Cabinda', 'provincia:Kwanza Norte',
        'tipo_credito:Crédito Habitação', 'tipo_credito:Pessoal',
        'sexo:Masculino', 'empreendedor:Sim', 'empreendedor:Não']

    input_train = x_train_with_ref_cat.drop(ref_categories,axis=1)

    reg = LogisticRegression()
    pd.options.display.max_rows = None
    reg.fit(input_train,y_train)
    reg.intercept_
    reg.coef_

    feature_name = input_train.columns.values

    #summary table
    summary_table = pd.DataFrame(columns = ['Feature name'], data = feature_name)
    summary_table['Coefficients'] = np.transpose(reg.coef_)
    summary_table.index = summary_table.index + 1
    summary_table.loc[0] = ['Intercept',reg.intercept_[0]]
    summary_table = summary_table.sort_index()
    summary_table

    #input test with probability
    inputs_test = x_train_with_ref_cat.drop(ref_categories, axis=1)
    inputs_test.head
    y_hat_test = reg.predict(inputs_test)
    y_hat_test_proba = reg.predict_proba(inputs_test)
    y_hat_test_proba[: ][: , 1]
    y_hat_test_proba = y_hat_test_proba[: ][: ,1]

    # x train test temp
    x_train_test_temp = x_train
    x_train_test_temp.reset_index(drop = True, inplace = True)
    df_actual_predicted_probs = pd.concat([x_train_test_temp, pd.DataFrame(y_hat_test_proba)],axis=1)
    return y_hat_test_proba

#Visualizing results
def plot_by_woe(df_WoE,rotation_of_x_axis_labels = 0):
    x = np.array(df_WoE.iloc[: , 0].apply(str))
    y = df_WoE['WoE']
    plt.figure(figsize = (18,6))
    plt.plot(x,y, marker = 'o', linestyle = '--', color = 'k')
    plt.xlabel(df_WoE.columns[0])
    plt.ylabel('Peso da Prova')
    plt.title(str('Peso da Prova por' + df_WoE.columns[0]))
    plt.xticks(rotation = rotation_of_x_axis_labels)



def woe_ordered_continuous(df,discrete_variable_name,good_bad_variable_df):
    df = pd.concat([df[discrete_variable_name],good_bad_variable_df], axis = 1)
    df = pd.concat([df.groupby(df.columns.values[0], as_index = False)[df.columns.values[1]].count(),
                 df.groupby(df.columns.values[0], as_index = False)[df.columns.values[1]].mean()], axis = 1)
    df = df.iloc[:, [0, 1, 3]]
    df.columns = [df.columns.values[0], 'n_obs','prop_n_bom']
    df['prop_n_obs'] = df['n_obs'] / df['n_obs'].sum()
    df['n_bom'] = df['prop_n_bom'] * df['n_obs']
    df['n_ruim'] = (1 - df['prop_n_bom']) * df['n_obs']
    df['prop_n_bom'] = df['n_bom'] / df['n_bom'].sum()
    df['prop_n_ruim'] = df['n_ruim'] / df['n_ruim'].sum()
    df['WoE'] = np.log(df['prop_n_bom'] / df['prop_n_ruim'])
    df['diff_prop_bom'] = df['prop_n_bom'].diff().abs()
    df['diff_WoE'] = df['WoE'].diff().abs()
    df['IV'] = (df['prop_n_bom'] - df['prop_n_bom'] * df['WoE'])
    df['IV'] = df['IV'].sum()
    return df

def woe_discrete(df,discrete_variable_name,good_bad_variable_df):
        df = pd.concat([df[discrete_variable_name],good_bad_variable_df], axis = 1)
        df = pd.concat([df.groupby(df.columns.values[0], as_index = False)[df.columns.values[1]].count(),
                    df.groupby(df.columns.values[0], as_index = False)[df.columns.values[1]].mean()], axis = 1)
        df = df.iloc[:, [0, 1, 3]]
        df.columns = [df.columns.values[0], 'n_obs','prop_bom']
        df['prop_n_obs'] = df['n_obs'] / df['n_obs'].sum()
        df['n_bom'] = df['prop_bom'] * df['n_obs']
        df['n_ruim'] = (1 - df['prop_bom']) * df['n_obs']
        df['prop_n_bom'] = df['n_bom'] / df['n_bom'].sum()
        df['prop_n_ruim'] = df['n_ruim'] / df['n_ruim'].sum()
        df['WoE'] = np.log(df['prop_n_bom'] / df['prop_n_ruim'])
        df = df.sort_values(['WoE'])
        df = df.reset_index(drop = True)
        df['diff_prop_bom'] = df['prop_bom'].diff().abs()
        df['diff_WoE'] = df['WoE'].diff().abs()
        df['IV'] = (df['prop_n_bom'] - df['prop_n_ruim'] * df['WoE'])
        df['IV'] = df['IV'].sum()
        return df





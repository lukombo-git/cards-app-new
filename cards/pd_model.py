
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn import metrics
import pandas as pd
import json
import requests
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

def loanData():
    #url = 'https://app-credit-cards.herokuapp.com/cards/cards_endpointclientes/'
    url = 'http://127.0.0.1:8000/cards/cards_endpointclientes/'
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

    return loan_data

def trainVariables():
    loan_data = loanData()
    #training the data
    x_train,x_test,y_train,y_test = train_test_split(loan_data.drop('bom_ruim',axis = 1), loan_data['bom_ruim'], test_size = 0.2)
    #Data preparation example
    df_input_prepr = x_train
    df_targets_prepr = y_train

    x_train['provincia:Cabinda'] = np.where((x_train['provincia'] == 'Cabinda'),1,0)
    x_train['provincia:Kwanza Norte'] = np.where((x_train['provincia'] == 'Kwanza Norte'),1,0)
    x_train['tipo_credito:Crédito Habitação'] = np.where((x_train['tipo_credito'] == 'Crédito Habitação'),1,0)
    x_train['tipo_credito:Pessoal'] = np.where((x_train['tipo_credito'] == 'Pessoal'),1,0)
    x_train['sexo:Masculino'] = np.where((x_train['sexo'] == 'Masculino'),1,0)
    x_train['sexo:Feminino'] = np.where((x_train['sexo'] == 'Feminino'),1,0)
    x_train['empreendedor:Sim'] = np.where((x_train['empreendedor'] == 'Sim'),1,0)
    x_train['empreendedor:Não'] = np.where((x_train['empreendedor'] == 'Não'),1,0)

    x_test['provincia:Cabinda'] = np.where((x_test['provincia'] == 'Cabinda'),1,0)
    x_test['provincia:Kwanza Norte'] = np.where((x_test['provincia'] == 'Kwanza Norte'),1,0)
    x_test['tipo_credito:Crédito Habitação'] = np.where((x_test['tipo_credito'] == 'Crédito Habitação'),1,0)
    x_test['tipo_credito:Pessoal'] = np.where((x_test['tipo_credito'] == 'Pessoal'),1,0)
    x_test['sexo:Masculino'] = np.where((x_test['sexo'] == 'Masculino'),1,0)
    x_test['sexo:Feminino'] = np.where((x_test['sexo'] == 'Feminino'),1,0)
    x_test['empreendedor:Sim'] = np.where((x_test['empreendedor'] == 'Sim'),1,0)
    x_test['empreendedor:Não'] = np.where((x_test['empreendedor'] == 'Não'),1,0)
    #training the data
    y_train[:2] = 1

    x_train_with_ref_cat = x_train.loc[:, ['provinciaCabinda',
        'provinciaKwanza Norte', 'tipo_creditoCrédito Habitação', 'tipo_creditoPessoal',
        'sexo:Masculino','sexo:Feminino','empreendedor:Sim','empreendedor:Não']]

    x_test_with_ref_cat = x_test.loc[:, ['provinciaCabinda',
        'provinciaKwanza Norte', 'tipo_creditoCrédito Habitação', 'tipo_creditoPessoal',
        'sexo:Masculino','sexo:Feminino', 'empreendedor:Sim','empreendedor:Não']]

    #relevante variable for a PD model
    ref_categories = ['sexo:Feminino','empreendedor:Não']
    input_train = x_train_with_ref_cat.drop(ref_categories,axis=1)
    #saving the training data
    input_train.to_csv('input_train.csv')

    input_test = x_test_with_ref_cat.drop(ref_categories,axis=1)
    input_test.to_csv('input_test.csv')

    feature_name = input_train.columns.values
    return feature_name,input_train,input_test,y_train,y_test,ref_categories,x_test_with_ref_cat

def regfunction():
    input_train = trainVariables()[1]
    y_train = trainVariables()[3]

    reg = LogisticRegression()
    pd.options.display.max_rows = None
    reg.fit(input_train,y_train)
    reg.intercept_
    reg.coef_

    return reg

def summaryTable():
    reg = regfunction()
    feature_name = trainVariables()[0]
    #summary table
    summary_table = pd.DataFrame(columns = ['Feature name'], data = feature_name)
    summary_table['Coefficientes'] = np.transpose(reg.coef_)
    summary_table.index = summary_table.index + 1
    summary_table.loc[0] = ['Intercept',reg.intercept_[0]]
    summary_table = summary_table.sort_index()
    return summary_table

    #************* Predicting the probability for single customer ****************************#
def actualPredictedProbs():
    reg = regfunction()
    input_test = trainVariables()[2]
    y_test = trainVariables()[4]
    y_hat_test = reg.predict(input_test)
    y_hat_test_proba = reg.predict_proba(input_test)
    y_hat_test_proba[: ][: , 1]
    y_hat_test_proba = y_hat_test_proba[: ][: ,1]
    # x train test temp
    x_train_test_temp = y_test
    x_train_test_temp.reset_index(drop = True, inplace = True)
    df_actual_predicted_probs = pd.concat([x_train_test_temp, pd.DataFrame(y_hat_test_proba)],axis=1)
    df_actual_predicted_probs.columns = ['y_test','y_hat_test_proba']
    df_actual_predicted_probs.index = y_test.index
    #Accuracy and Area under the curve
    tr = 0.9
    df_actual_predicted_probs['y_hat_test'] = np.where(df_actual_predicted_probs['y_hat_test_proba'] > tr,1,0)
    #metric auroc
    df_actual_predicted_probs['y_test'][:1] = 1
    AUROC = roc_auc_score(df_actual_predicted_probs['y_test'],df_actual_predicted_probs['y_hat_test_proba'])

    df_actual_predicted_probs = df_actual_predicted_probs.sort_values('y_hat_test_proba')
    df_actual_predicted_probs = df_actual_predicted_probs.reset_index()
    
    df_actual_predicted_probs['População N cumulativa'] = df_actual_predicted_probs.index + 1
    df_actual_predicted_probs['Cumulativo N bom'] = df_actual_predicted_probs['y_test'].cumsum()
    df_actual_predicted_probs['Cumulativo N ruim'] = df_actual_predicted_probs['População N cumulativa'] - df_actual_predicted_probs['Cumulativo N bom'].cumsum() 
    

    df_actual_predicted_probs['População Perc cumulativa'] = df_actual_predicted_probs['População N cumulativa'] / (df_actual_predicted_probs.shape[0])
    df_actual_predicted_probs['Perc cumulativo bom'] = df_actual_predicted_probs['Cumulativo N bom'] / (df_actual_predicted_probs['y_test'].sum())
    df_actual_predicted_probs['Perc cumulativo ruim'] = df_actual_predicted_probs['Cumulativo N ruim'] / (df_actual_predicted_probs.shape[0] - df_actual_predicted_probs['y_test'].sum())
    #Gini = AUROC * 2 - 1
    #KS = max(df_actual_predicted_probs['Perc cumulativo ruim'] - df_actual_predicted_probs['Perc cumulativo bom'])
    #print(KS)
    return df_actual_predicted_probs   
    
def scoreCard():
    #Creating a score card
    summary_table = summaryTable()
    ref_categories = trainVariables()[5]
    df_ref_categories = pd.DataFrame(columns = ['Feature name'], data = ref_categories)
    df_ref_categories['Coefficientes'] = 0
    df_ref_categories['p_values'] = np.nan
    df_scorecard = pd.concat([summary_table, df_ref_categories])
    df_scorecard = df_scorecard.reset_index()
    df_scorecard['Nome Original da feature'] = df_scorecard['Feature name'].str.split(':').str[0]

    # min and max score
    min_score = 300
    max_score = 850

    # min and max coef
    min_sum_coef = df_scorecard.groupby('Nome Original da feature')['Coefficientes'].min().sum()
    max_sum_coef = df_scorecard.groupby('Nome Original da feature')['Coefficientes'].max().sum()

    # df score calculation
    df_scorecard['Pontuação - Cálculo'] = (df_scorecard['Coefficientes'] * (max_score - min_score) / (max_sum_coef - min_sum_coef))
    df_scorecard['Pontuação - Cálculo'][0] = ((df_scorecard['Coefficientes'][0] - min_sum_coef)/ (max_sum_coef - min_sum_coef) * (max_score - min_score)) 

    # df score preliminary
    df_scorecard['Pontuação - Preliminar'] = df_scorecard['Pontuação - Cálculo'].round()

    # min sum score prel
    min_sum_score_prel = df_scorecard.groupby('Nome Original da feature')['Pontuação - Preliminar'].min().sum()
   
    # max sum score prel
    max_sum_score_prel = df_scorecard.groupby('Nome Original da feature')['Pontuação - Preliminar'].max().sum()

    df_scorecard['Diferença'] = df_scorecard['Pontuação - Preliminar'] - df_scorecard['Pontuação - Cálculo']
    df_scorecard['Pontuação - final'] = df_scorecard['Pontuação - Preliminar']
    #df_scorecard['Pontuação - final'][77] = 16

    #score Repeting to see if the min and max score are the same# 
    # min sum score prel
    min_sum_score_prel = df_scorecard.groupby('Nome Original da feature')['Pontuação - Preliminar'].min().sum()
    # max sum score prel
    max_sum_score_prel = df_scorecard.groupby('Nome Original da feature')['Pontuação - Preliminar'].max().sum()
    #dropping the feature
    df_scorecard.drop('Nome Original da feature',axis=1,inplace=True)
    return df_scorecard,min_score,max_score,max_sum_coef,min_sum_coef,df_scorecard['Feature name'].values,df_scorecard['Pontuação - final']

def interceptFeatureNames():
    df_scorecard = scoreCard()[5] #df_scorecard['Feature name']
    x_test_with_ref_cat = trainVariables()[6]
    x_test_with_ref_cat_w_intercept = x_test_with_ref_cat
    x_test_with_ref_cat_w_intercept.insert(0, 'Intercept', 1)
    x_test_with_ref_cat_w_intercept = x_test_with_ref_cat_w_intercept[df_scorecard]
    return x_test_with_ref_cat_w_intercept


def creditScore():
    #calculating credit scores
    x_test_with_ref_cat_w_intercept = interceptFeatureNames()
    df_scorecard = scoreCard()[6] #df_scorecard['Pontuação - Final]
    scorecard_scores = df_scorecard
    scorecard_scores = scorecard_scores.values.reshape(9, 1)
    y_scores = x_test_with_ref_cat_w_intercept.dot(scorecard_scores)
    return y_scores

def creditScoreToPD(): 
    y_scores = creditScore()
    min_score = scoreCard()[1] #min score
    max_score = scoreCard()[2] #max score
    max_sum_coef = scoreCard()[3] #max sum coef
    min_sum_coef = scoreCard()[4] #min sum coef
    
    sum_coef_from_score = ((y_scores - min_score) / (max_score - min_score)) * (max_sum_coef - min_sum_coef) + min_sum_coef
    y_hat_proba_from_score = np.exp(sum_coef_from_score) / (np.exp(sum_coef_from_score) + 1)
    
    return y_hat_proba_from_score


def cutOffs():
    df_actual_predicted_probs = actualPredictedProbs()
    fpr,tpr,thresholds = roc_curve(df_actual_predicted_probs['y_test'],df_actual_predicted_probs['y_hat_test_proba'])

    y_scores = creditScore()
    min_score = scoreCard()[1] #min score
    max_score = scoreCard()[2] #max score
    max_sum_coef = scoreCard()[3] #max sum coef
    min_sum_coef = scoreCard()[4] #min sum coef

    #Cute-off rate = used to for taking a decision wheter to approve a loan or not
    df_cutoffs = pd.concat([pd.DataFrame(thresholds),pd.DataFrame(fpr), pd.DataFrame(tpr)], axis = 1)
    df_cutoffs.columns = ['limiares','fpr','tpr']
    df_cutoffs['limiares'][0] = 1 - 1 / np.power(10, 16)
    df_cutoffs['Pontuação'] = ((np.log(df_cutoffs['limiares'] / (1 - df_cutoffs['limiares'])) - min_sum_coef) * ((max_score - min_score) / (max_sum_coef - min_sum_coef)) + min_score).round()
    df_cutoffs['Pontuação'][0] = max_score

    def n_approved(p):
        return np.where(df_actual_predicted_probs['y_hat_test_proba'] >= p, 1, 0).sum()
    
    df_cutoffs['N aprovado'] = df_cutoffs['limiares'].apply(n_approved)
    df_cutoffs['N rejeitado'] = df_actual_predicted_probs['y_hat_test_proba'].shape[0] - df_cutoffs['N aprovado']
    df_cutoffs['Taxa de aprovação'] = df_cutoffs['N aprovado'] / df_actual_predicted_probs['y_hat_test_proba'].shape[0]
    df_cutoffs['Taxa de rejeição'] = 1 - df_cutoffs['Taxa de aprovação']  
    return df_cutoffs



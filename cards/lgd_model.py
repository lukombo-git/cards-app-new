import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
#Build a Logistic Regression Model with P-Values
from sklearn import linear_model
import scipy.stats as stat
import pickle
import json
import requests

class LogisticRegression_with_p_values:
    def __init__(self,*args,**kwargs):
        self.model = linear_model.LogisticRegression(*args,**kwargs)
    def fit(self,X,y):
        denom = (2.0 * (1.0 + np.cosh(self.model.decision_function(X))))
        F_ij = np.dot((X /denom).T,X)
        Cramer_Rao = np.linalg.inv(F_i)
        sigma_estimates = np.sqrt(np.diagonal(Cramer_Rao))
        z_scores = self.model.coef_[0] / sigma_estimates
        p_values = [stat.norm.sf(abs(x)) * 2 for x in z_scores]
        self.coef_ = self.model.coef_
        selff.intercept_ = self.model.intercept_
        self.p_values = p_values

def loanData2():
   url = 'http://127.0.0.1:8000/cards/cards_endpointclientes/'
   r = requests.get(url)
   files = r.json()
   loan_data = pd.DataFrame(files)
   
   #loading the data
   loan_data_defaults = pd.DataFrame(loan_data['estado_credito'].isin(['pago','nao pago']))
   #recuperação_bruta_pós-baixa
   loan_data_defaults['taxa_de_recuperacao'] = loan_data['recoveries'] / loan_data['valor_financiamento']
   loan_data_defaults['taxa_de_recuperacao'] = np.where(loan_data_defaults['taxa_de_recuperacao'] < 1, 1, loan_data_defaults['taxa_de_recuperacao'])
   loan_data_defaults['taxa_de_recuperacao'] = np.where(loan_data_defaults['taxa_de_recuperacao'] < 0, 0, loan_data_defaults['taxa_de_recuperacao'])
   
   loan_data_defaults['CCF'] = (loan_data['valor_financiamento'] - loan_data['total_rec_prncp'] / loan_data['valor_financiamento'])
   loan_data_defaults['recovered_0_1'] = np.where(loan_data_defaults['taxa_de_recuperacao'] == 0,0,1)

   #Training the model
   x_train,x_test,y_train,y_test = train_test_split(loan_data_defaults.drop(['taxa_de_recuperacao','recovered_0_1','CCF'],axis = 1),loan_data_defaults['recovered_0_1'],test_size = 0.2, random_state = 42)
   y_train[:3] = 0

   #feature_all = get all the created dummy variables
   #feature_ref_cat = get the reference variable
   lgd_train_1 = pd.read_csv('input_train.csv')

   return loan_data_defaults,y_train,y_test,x_train,x_test,lgd_train_1

def regFunction():
   y_train = loanData2()[1]
   #Appling the model
   lgd_train_1= loanData2()[5]
   reg = LogisticRegression() #Regression with p_values
   #reg = LogisticRegression_with_p_values()
   reg.fit(lgd_train_1,y_train)
   return reg

def lgdSumTable2():
   lgd_train_1 = loanData2()[5]
   reg = regFunction()
   #getting the feature names
   feature_name = lgd_train_1.columns.values

   #creating the summary table
   summary_table = pd.DataFrame(columns = ['Feature name'], data = feature_name)
   summary_table['Coefficientes'] = np.transpose(reg.coef_)
   summary_table.index = summary_table.index + 1
   summary_table.loc[0] = ['Intercept',reg.intercept_[0]]
   summary_table = summary_table.sort_index()

   #p_values = reg.p_values
   #p_values = np.append(np.nan,np.array(p_values))
   #summary_table['p_values'] = p_values
   return  summary_table

def probaFunction():
   lgd_train_1= loanData2()[5]
   reg = regFunction()
   # getting data for test
   #lgd_x_test = x_test[features_all]
   #lgd_x_test = lgd_x_test.drop(features_reference_cat, axis = 1)
   lgd_x_test = lgd_train_1
   #making the probability of single customer
   y_hat_test = reg.predict(lgd_x_test)
   y_hat_test_proba = reg.predict_proba(lgd_x_test)
   y_hat_test_proba[: ][: , 1]
   y_hat_test_proba = y_hat_test_proba[: ][: ,1]

   #saving the probability
   proba_y = pd.DataFrame(y_hat_test_proba)
   proba_y.columns =['lgd_proba']
   proba_y.to_csv('lgd_proba.csv')

   return proba_y, lgd_x_test

def lgdActualPreditedProbs():
    #LGD MODEL estimation and accuracy of the model
    y_hat_test_proba = probaFunction()[0]
    x_train_test_temp = loanData2()[2] #y_test
    lgd_x_test = probaFunction()[1]
    reg = regFunction() 
    x_train_test_temp.reset_index(drop = True, inplace = True)
    df_actual_predicted_probs = pd.concat([x_train_test_temp, pd.DataFrame(y_hat_test_proba)],axis=1)
    df_actual_predicted_probs.columns = ['lgd_x_test','y_hat_test_proba_lgd']
    df_actual_predicted_probs.index = lgd_x_test.index
    df_actual_predicted_probs['lgd_x_test'].fillna(0,inplace=True)
    
    tr = 0.5
    fpr,tpr,thresholds = roc_curve(df_actual_predicted_probs['lgd_x_test'],df_actual_predicted_probs['y_hat_test_proba_lgd'])    
    df_actual_predicted_probs['y_hat_test_lgd'] = np.where(df_actual_predicted_probs['y_hat_test_proba_lgd'] > tr,1,0)
    #pd.crosstab(df_actual_predicted_probs['lgd_x_test'],df_actual_predicted_probs['y_hat_test_proba_lgd_stage_1'],rownames = ['Actual'], colnames= ['Predicted'])
    #pd.crosstab(df_actual_predicted_probs['x_test'],df_actual_predicted_probs['y_hat_test'],rownames = ['Actual'], colnames= ['Predicted']) / df_actual_predicted_probs.shape[0]
    #pd.crosstab(df_actual_predicted_probs['x_test'],df_actual_predicted_probs['y_hat_test'],rownames = ['Actual'], colnames= ['Predicted']) / df_actual_predicted_probs.shape[0]).iloc[1,1]
    return df_actual_predicted_probs

def LgdMetrics():
    AUROC = roc_auc_score(df_actual_predicted_probs['lgd_x_test'],df_actual_predicted_probs['y_hat_test_proba_lgd'])
    #saving the training data
    loan_data_defaults.to_csv('lgd_model.csv')
    #saving the model
    #pickle.dump(lgd_train_1,open('lgd_train_1.sav','wb'))
    return AUROC
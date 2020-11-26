
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import pickle

def getExpectedLost():
        #loading the data
        loan_data_preprocessed = pd.read_csv('lgd_model.csv')

        #Appling the model
        reg = LogisticRegression()
        #reg.fit(lgd_train_1,y_train)

        loan_data_preprocessed_lgd_ead = pd.read_csv('input_train.csv')

        #loan_data_preprocessed_lgd_ead = loan_data_preprocessed[features_all]
        #loan_data_preprocessed_lgd_ead = ead_x_test.drop(features_reference_cat, axis = 1)

        #loan_data_preprocessed['recovery_rate_st_1'] = reg.predict(loan_data_preprocessed_lgd_ead) 
        loan_data_preprocessed['recovery_rate'] = loan_data_preprocessed['taxa_de_recuperacao'] * loan_data_preprocessed['taxa_de_recuperacao']
        loan_data_preprocessed['recovery_rate'] = np.where(loan_data_preprocessed['taxa_de_recuperacao'] < 0, 0, loan_data_preprocessed['taxa_de_recuperacao'])
        loan_data_preprocessed['recovery_rate'] = np.where(loan_data_preprocessed['taxa_de_recuperacao'] < 1, 1, loan_data_preprocessed['taxa_de_recuperacao'])

        #Calculating LGD
        loan_data_preprocessed['LGD'] = 1 - loan_data_preprocessed['recovery_rate']

        #loan_data_preprocessed['CCF'] = reg_ead.predict(loan_data_preprocessed_lgd_ead)

        loan_data_preprocessed['CCF'] = np.where(loan_data_preprocessed['CCF'] < 0, 0, loan_data_preprocessed['CCF'])
        loan_data_preprocessed['CCF'] = np.where(loan_data_preprocessed['CCF'] < 1, 1, loan_data_preprocessed['CCF'])

        loan_data_train = pd.read_csv('input_train.csv')
        loan_data_test = pd.read_csv('input_test.csv')
        loan_data_train_pd = pd.concat([loan_data_train,loan_data_test], axis = 0)

        #feature_all_pd = ['provinciaCabinda',
        #        'provinciaKwanza Norte', 'provinciaLuanda', 'provinciaLuanda Norte',
        #        'tipo_creditoCrédito Habitação', 'tipo_creditoPessoal',
        #        'empreendedorNão', 'empreendedorSim', 'como_quer_pagarPor Prestações',
        #        'como_quer_pagarPor prestação', 'como_quer_pagarValor Completo',
        #        'provincia:Cabinda', 'provincia:Kwanza Norte',
        #        'tipo_credito:Crédito Habitação', 'tipo_credito:Pessoal',
        #        'sexo:Masculino', 'empreendedor:Sim', 'empreendedor:Não']

        #relevante variable for a PD model
        #ref_categories = [
        #        'provincia:Cabinda', 'provincia:Kwanza Norte',
        #        'tipo_credito:Crédito Habitação', 'tipo_credito:Pessoal',
        #        'sexo:Masculino', 'empreendedor:Sim', 'empreendedor:Não']

        #loan_data_train_pd_temp = loan_data_train[features_all_pd]
        #loan_data_train_pd_temp = loan_data_train_pd_temp.drop(features_reference_cat, axis = 1)


        #reg_pd = pickle.load(open('pd_model.sav','rb'))

        loan_data_preprocessed_new = pd.concat([loan_data_preprocessed, loan_data_train], axis=1)

        #loan_data_preprocessed_new['EL'] = loan_data_preprocessed_new['PD'] * loan_data_preprocessed_new['LGD'] * loan_data_preprocessed_new['EAD']
        #loan_data_preprocessed_new[['funded_amnt','PD','LGD','EL']]
        #loan_data_preprocessed_new['EL'].sum()
        #loan_data_preprocessed_new['funded_amnt'].sum()
        #loan_data_preprocessed_new['EL'].sum() / loan_data_preprocessed_new['funded_amnt'].sum()

        #loan_data_preprocessed_new['EL'] = loan_data_preprocessed_new['LGD'] * loan_data_preprocessed_new['LGD'] * loan_data_preprocessed_new['LGD']
        loan_data_preprocessed['EL'] = loan_data_preprocessed_new['LGD'] * loan_data_preprocessed_new['LGD'] * loan_data_preprocessed_new['LGD']

        return loan_data_preprocessed

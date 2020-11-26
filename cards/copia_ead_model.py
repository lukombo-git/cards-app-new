import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import roc_curve, roc_auc_score
#Build a Logistic Regression Model with P-Values
from sklearn import linear_model
import scipy.stats as stat
import pickle
from sklearn import linear_model
import scipy.stats as stat

class LinearRegression(linear_model.LinearRegression):
    def __init__(self, fit_intercept=True, normalize=False, copy_X=True, n_jobs=1):
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.copy_X = copy_X
        self.n_jobs = n_jobs
    def fit(self,X,y, n_jobs=1):
        self = super(LinearRegression, self).fit(X,y, n_jobs)
        sse = np.sum((self.predict(X) - y) ** 2, axis = 0) / float(X.shape[0] - X.shape[1])
        se = np.array([np.sqrt(np.diagonal(sse * np.linalg.inv(np.dot(X.T, X))))])
        self.t = self.coef_ / se
        sel.p = np.squeeze(2 * (1- stat.t.cdf(np.abs(self.t), y.shape[0] - X.shape[1])))
        return self

#loading the data
loan_data_defaults = pd.read_csv('lgd_model.csv')

lgd_stage_2_data = loan_data_defaults[loan_data_defaults['recovery_rate_0_1'] == 1]

#training the data
x_train_stage2,x_test_stage2,y_train_stage_2,y_test_stage2 = train_test_split(loan_data_defaults.drop(['good_bad','recovery_rate','recovery_rate_0_1','CCF'],axis = 1),loan_data_defaults['recovery_rate_0_1'],test_size = 0.2, random_state =42)

#dropping all features
#lgd_x_train_stage2 = x_test[features_all]
#lgd_x_train_stage2 = lgd_x_train_stage2.drop(features_reference_cat, axis = 1)
lgd_x_test = loan_data_defaults

#Appling the model
reg_lgd2 = LinearRegression()
reg_lgd2.fit(x_train_stage2,y_train_stage_2)

# feature name
feature_name = x_train_stage2.columns.values

#creating the summary table
summary_table = pd.DataFrame(columns = ['Feature name'], data = feature_name)
summary_table['Coefficients'] = np.transpose(reg_lgd2 .coef_)
summary_table.index = summary_table.index + 1
summary_table.loc[0] = ['Intercept',reg_lgd2 .intercept_[0]]
summary_table = summary_table.sort_index()
p_values = reg.p_values
p_values = np.append(np.nan,np.array(p_values))
summary_table['p_values'] = p_values

#testing the data
lgd_x_test2 = x_test[features_all]
lgd_x_test2 = lgd_x_test.drop(features_reference_cat, axis = 1)
lgd_x_test2.columns.values

#making the probability
y_hat_test2 = reg_lgd2.predict(lgd_x_test2)
lgd_x_test2_temp = lgd_x_test2
lgd_x_test2_temp = lgd_x_test2_temp.reset_index(drop = True)
df_actual_predicted_probs = pd.concat([lgd_x_test2_temp, pd.DataFrame(y_hat_test2)],axis=1).corr()

#saving the model
pickle.dump(reg_lgd2, open('lgd_model_stage2.sav','wb'))

y_hat_test_lgd_stage_2_all = reg_lgd2.predict(lgd_x_test)
y_hat_test_lgd = y_hat_test * y_hat_test2
y_hat_test_lgd = np.where(y_hat_test_lgd < 0, 0, y_hat_test_lgd)
y_hat_test_lgd = np.where(y_hat_test_lgd < 1, 1, y_hat_test_lgd)

#training the data
ead_x_train,ead_x_test,ead_y_train,ead_y_test = train_test_split(loan_data_defaults.drop(['good_bad','recovery_rate','recovery_rate_0_1','CCF'],axis = 1),loan_data_defaults['recovery_rate_0_1'],test_size = 0.2, random_state =42)

#dropping all features
ead_x_train = ead_x_train[feature_all]
ead_x_train = ead_x_train.drop(features_reference_cat, axis = 1)

reg_ead = LinearRegression()
reg_ead.fit(ead_x_train,ead_y_train)

feature_name = ead_x_train.columns.values
summary_table = pd.DataFrame(columns = ['Feature name'], data = feature_name)
summary_table['Coefficients'] = np.transpose(reg_ead .coef_)
summary_table.index = summary_table.index + 1
summary_table.loc[0] = ['Intercept',reg_ead .intercept_[0]]
summary_table = summary_table.sort_index()
p_values = reg_ead.p_values
p_values = np.append(np.nan,np.array(p_values))
summary_table['p_values'] = p_values

#testing the data
ead_x_test = ead_x_test[feature_all]
ead_x_test = ead_x_test.drop(features_reference_cat, axis = 1)

#probability
y_hat_test_ead = reg_ead.predict(ead_x_test)
ead_x_test_temp = ead_x_test
ead_x_test_temp = y_hat_test_temp.reset_index(drop = True)
pd.concat([ead_x_test_temp,pd.DataFrame(y_hat_test_ead)], axis = 1).corr()


y_hat_test_ead = np.where(y_hat_test_lgd < 0, 0, y_hat_test_ead)
y_hat_test_ead = np.where(y_hat_test_lgd < 1, 1, y_hat_test_ead)
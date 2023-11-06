import sys
import pandas as pd
import numpy as np
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error

#submit base file name
file_input = sys.argv[1]

#sklearn Bayesian
base_regressor = BayesianRidge()
log_regressor = BayesianRidge()
regressor = BayesianRidge()

#File Names from Base Name
train_suffix = '_train.csv'
validate_suffix = '_validate.csv'
test_suffix = '_test.csv'

#Read Data from File
df_train = pd.read_csv(file_input + train_suffix)
df_validate = pd.read_csv(file_input + validate_suffix)
df_test = pd.read_csv(file_input + test_suffix)

#Get the Features X into one Matrix and extract what we're looking for
train_x, train_y, log_train_y = df_train.drop(columns=['Amount','log_Amount']), df_train['Amount'], df_train['log_Amount']

#Finding Bayesian Hyperparameters
bayesian_param = {
    'alpha_1': np.linspace(1e-2,5,num=100),
    'alpha_2': np.linspace(1e-2,5,num=100),
    'lambda_1': np.linspace(1e-2,5,num=100),
    'lambda_2':np.linspace(1e-2,5,num=100)
}

#Randomized Search to find the best hyperparameters
log_random_search_cv = RandomizedSearchCV(log_regressor, param_distributions=bayesian_param,n_iter=100, cv=10, verbose=1,n_jobs=5,random_state=1)
log_random_search_cv.fit(train_x,log_train_y)
log_param = log_random_search_cv.best_params_

#Randomized Search to find the best hyperparameters
random_search_cv = RandomizedSearchCV(regressor, param_distributions=bayesian_param,n_iter=100, cv=10, verbose=1,n_jobs=5,random_state=1)
random_search_cv.fit(train_x,train_y)
param = random_search_cv.best_params_

#Base hyperparameters
base_regressor.fit(train_x,train_y)

#Log model with hyperparameters
log_model = BayesianRidge(alpha_1 = log_param['alpha_1'], alpha_2 = log_param['alpha_2'], lambda_1 = log_param['lambda_1'], lambda_2 = log_param['lambda_2'])
log_model.fit(train_x, log_train_y)

#Model with hyperparamters
trained_model = BayesianRidge(alpha_1 = param['alpha_1'], alpha_2 = param['alpha_2'], lambda_1 = param['lambda_1'], lambda_2 = param['lambda_2'])
trained_model.fit(train_x,train_y)

#Extract Data from the Validation data set to train the hyperparameter
validate_x, validate_y, log_validate_y = df_validate.drop(columns=['Amount','log_Amount']), df_validate['Amount'], df_validate['log_Amount']

#Here we compare validate our model to see which one is the best in this dataset.
base_predict_validate = base_regressor.predict(validate_x)
log_predict_validate = log_model.predict(validate_x)
tuned_predict_validate = trained_model.predict(validate_x)

base_predict_data = {'validate_y':validate_y,'base_predict':base_predict_validate}
df_base_predict = pd.DataFrame(base_predict_data)

log_predict_data = {'log_validate_y':log_validate_y,'base_predict':log_predict_validate}
df_log_predict = pd.DataFrame(log_predict_data)

tuned_predict_data = {'validate_y':validate_y,'base_predict':tuned_predict_validate}
df_tuned_predict = pd.DataFrame(tuned_predict_data)

df_base_predict.to_csv(file_input + '_base_validate.csv', index=False)
print("Successfully written data to: " + file_input + '_base_validate.csv')
df_log_predict.to_csv(file_input + '_log_validate.csv', index=False)
print("Successfully written data to: " + file_input + '_log_validate.csv')
tuned_predict_data.to_csv(file_input + '_tuned_validate.csv', index =False)
print("Successfully written data to: " + file_input + '_tuned_validate.csv')
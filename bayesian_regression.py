import sys
import pandas as pd
import numpy as np
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import RandomizedSearchCV

#submit base file name
file_input = sys.argv[1]

#sklearn Bayesian
base_regressor = BayesianRidge()
log_base_regressor = BayesianRidge()

tuned_regressor = BayesianRidge()
log_tuned_regressor = BayesianRidge()

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
    'alpha_1': np.linspace(1e-2,4,num=100),
    'alpha_2': np.linspace(1e-2,4,num=100),
    'lambda_1': np.linspace(1e-2,4,num=100),
    'lambda_2':np.linspace(1e-2,4,num=100)
}

#Randomized Search to find the best hyperparameters
base_regressor.fit(train_x,train_y)
log_base_regressor.fit(train_x, log_train_y)

#Randomized Search to find the best hyperparameters
tuned_cv = RandomizedSearchCV(tuned_regressor, param_distributions=bayesian_param,n_iter=100, cv=10, verbose=1,n_jobs=10,random_state=1)
tuned_cv.fit(train_x,train_y)
tuned_param = tuned_cv.best_params_

log_tuned_cv = RandomizedSearchCV(log_tuned_regressor, param_distributions=bayesian_param,n_iter=100, cv=10, verbose=1,n_jobs=10,random_state=1)
log_tuned_cv.fit(train_x,log_train_y)
log_tuned_param = log_tuned_cv.best_params_

#Model with hyperparamters
tuned_model = BayesianRidge(alpha_1 = tuned_param['alpha_1'], alpha_2 = tuned_param['alpha_2'], lambda_1 = tuned_param['lambda_1'], lambda_2 = tuned_param['lambda_2'])
tuned_model.fit(train_x,train_y)

#Log model with hyperparameters
log_tuned_model = BayesianRidge(alpha_1 = log_tuned_param['alpha_1'], alpha_2 = log_tuned_param['alpha_2'], lambda_1 = log_tuned_param['lambda_1'], lambda_2 = log_tuned_param['lambda_2'])
log_tuned_model.fit(train_x, log_train_y)

#Extract Data from the Validation data set to train the hyperparameter
validate_x, validate_y, log_validate_y = df_validate.drop(columns=['Amount','log_Amount']), df_validate['Amount'], df_validate['log_Amount']

#Here we compare validate our model to see which one is the best in this dataset.
base_predict_validate = base_regressor.predict(validate_x)
log_predict_validate = log_base_regressor.predict(validate_x)
tuned_predict_validate = tuned_model.predict(validate_x)
log_tuned_predict_validate = log_tuned_model.predict(validate_x)

base_predict_data = {'validate_y':validate_y,'base_predict':base_predict_validate}
df_base_predict = pd.DataFrame(base_predict_data)

log_predict_data = {'log_validate_y':log_validate_y,'base_predict':log_predict_validate}
df_log_base_predict = pd.DataFrame(log_predict_data)

tuned_predict_data = {'validate_y':validate_y,'base_predict':tuned_predict_validate}
df_tuned_predict = pd.DataFrame(tuned_predict_data)

log_tuned_predict_data = {'validate_y':validate_y,'base_predict':log_tuned_predict_validate}
df_log_tuned_predict = pd.DataFrame(log_tuned_predict_data)


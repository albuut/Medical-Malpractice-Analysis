import sys
import pandas as pd
import numpy as np
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_squared_error

#submit base file name
file_input = sys.argv[1]

#SKLEARN Bayesian
log_regressor = BayesianRidge()
regressor = BayesianRidge()

#File Names from Base Name
train_suffix = '_train.csv'
validate_suffix = '_validate.csv'
test_suffix = '_test.csv'

#Read Data from File
df_train = pd.read_csv(file_input + train_suffix)
df_validate = pd.read_csv(file_input + validate_suffix)
df_test = pd.read_Csv(file_input + test_suffix)

#Get the Features X into one Matrix and extract what we're looking for
train_x = df_train.drop(columns=['Amount','log_Amount'])

#Get the target feature we want to predict
log_train_y = df_train['log_Amount']
train_y = df_train['Amount']

#Check with log transform Transform
log_regressor.fit(train_x, log_train_y)
#Check without log Transform
regressor.fit(train_x, train_y)

##TODO
# Use current model to adjust hyperparameter and test with the validation data.
# Look for methods to adjust hyperparameter
# Checkout https://towardsdatascience.com/a-conceptual-explanation-of-bayesian-model-based-hyperparameter-optimization-for-machine-learning-b8172278050f



#Extract Data from the Validation data set to train the hyperparameter
validate_x = df_test.drop(columns=['Amount','log_Amount'])
log_validate_y = df_test['log_Amount']
validate_y = df_test['Amount']

#Possibly a loop here or some kind of method to go about training the data
log_y_predict = log_regressor.predict(validate_X)
y_predict = regressor.predict(validate_x)

#Figure out a good method of evaluation of the amount
#Checkout https://stats.stackexchange.com/questions/51046/how-to-check-if-my-regression-model-is-good
log_mse = mean_squared_error(validate_y, log_y_predict)
mse = mean_squared_error(validate_y, y_predict)


##TODO
#Use the new hyperparameter data for newly trained model
#Use the test data to retrieve points to evaluate
#Write data to csv to use for analysis later 
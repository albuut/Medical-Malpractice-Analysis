import sys
import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
from stepwise_regression import step_reg
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

# input prprocessed .csv malpractice data file in from the command line
file_input = sys.argv[1]

# read in inputted file

train_suffix = '_train.csv'
validate_suffix = '_validate.csv'
test_suffix = '_test.csv'

df = pd.read_csv(file_input + train_suffix)
df_validate = pd.read_csv(file_input + validate_suffix)
df_test = pd.read_csv(file_input + test_suffix)

# consider all features (independent variables) except for 'Amount' and 'log_Amount'
X = df.drop(['log_Amount', 'Amount'], axis=1)

# 'log_Amount' is the dependent variable here
# the goal is to see which independent variables signficantly impact the dependent variable
y = df['log_Amount']

# add a constant to intercept X
X = sm.add_constant(X)

# create a linear regression model using our variables
model = sm.OLS(y, X)
fit_model = model.fit()
# predict using all independent variables as features
baseline_predict = fit_model.predict(X)
df_test_y = df_test['log_Amount']
baseline_predict = baseline_predict[:len(df_test_y)]
# compute rmse and msa of baseline model predictions
baseline_mse = mean_squared_error(df_test_y, baseline_predict)
baseline_mae = mean_absolute_error(df_test_y, baseline_predict)
# displays the results of the model
# print(fit_model.summary())
print("Baseline selection RMSE:", np.sqrt(baseline_mse))
print("Baseline selection MAE:", baseline_mae)


# using the step_reg library, perform backward selection on the features (X)
# backselect will store a list of feature names sorted by significance to 'log_Amount' (highest to lowest)
backselect = step_reg.backward_regression(X, y, 0.05, verbose=False)
# print(backselect)

# now create a new model with the backward selected features
# ensure that predictors are significant (i.e p-value >0.05)
X_backselect = X[backselect]
back_model = sm.OLS(y, X_backselect)
fit_back_model = back_model.fit()
# display the results of the model
# print(fit_back_model.summary())

# make predictions using the new model with selected features from backward selection
b_y_predict = fit_back_model.predict(X_backselect)
# use the testing set
b_y_predict = b_y_predict[:len(df_test_y)]
# calculate the root mean squared error and mean absolute error, t-statistic, and p-value of the predictions compared to the actual values
b_mse = mean_squared_error(df_test_y, b_y_predict)
b_mae = mean_absolute_error(df_test_y, b_y_predict)
t_stat_backward, p_val_backward = stats.ttest_ind(
    df_test['log_Amount'], b_y_predict)

print("\nBackward selection metrics:")
print("RMSE:", np.sqrt(b_mse))
print("MAE:", b_mae)
print("T-Statistic:", t_stat_backward)
print("P-Value:", p_val_backward)

# using the step_reg library, perform forward selection on the features (X)
# forwardselect will store a list of feature names sorted by significance to 'log_Amount' (highest to lowest)
forwardselect = step_reg.forward_regression(X, y, 0.05, verbose=False)
# print(forwardselect)

# now create a new model with the forward selected features
# ensure that predictors are significant (i.e p-value >0.05)
X_forwardselect = X[forwardselect]
forward_model = sm.OLS(y, X_forwardselect)
fit_forward_model = forward_model.fit()
# display the results of the model
# print(fit_forward_model.summary())

# make predictions using the new model with selected features from forward selection
f_y_predict = fit_forward_model.predict(X_forwardselect)
f_y_predict = f_y_predict[:len(df_test_y)]
# calculate the root mean squared error mean absolute error, t-statistic, and p-value of the predictions compared to the actual values
f_mse = mean_squared_error(df_test_y, f_y_predict)
f_mae = mean_absolute_error(df_test_y, f_y_predict)
t_stat_forward, p_val_forward = stats.ttest_ind(
    df_test['log_Amount'], f_y_predict)

print("\nForward selection metrics: ")
print("RMSE:", np.sqrt(f_mse))
print("MAE:", f_mae)
print("T-Statistic:", t_stat_forward)
print("P-Value:", p_val_forward)

# Scatter plot for backward selection
plt.figure(figsize=(10, 6))
plt.scatter(df_test['log_Amount'], b_y_predict,
            label='Backward Selection', alpha=0.7)
plt.xlabel('Actual log_Amount')
plt.ylabel('Predicted log_Amount')
plt.title('Scatter Plot for Backward Selection Model Predictions on Test Set')
plt.legend()
plt.show()

# Scatter plot for forward selection
plt.figure(figsize=(10, 6))
plt.scatter(df_test['log_Amount'], f_y_predict,
            label='Forward Selection', alpha=0.7)
plt.xlabel('Actual log_Amount')
plt.ylabel('Predicted log_Amount')
plt.title('Scatter Plot for Forward Selection Model Predictions on Test Set')
plt.legend()
plt.show()

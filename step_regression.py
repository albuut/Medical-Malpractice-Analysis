import sys
import pandas as pd
import numpy as np
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

# Create data frame will all the data for k-fold cross validation
file_paths = [file_input + train_suffix, file_input +
              validate_suffix, file_input + test_suffix]
# Create an empty DataFrame to store the combined data
combined_data = pd.DataFrame()
# Iterate through each file and concatenate its data to the combined DataFrame
for file_path in file_paths:
    combined_data = pd.concat([combined_data, df], ignore_index=True)
# Store features and target variable
X_total = combined_data.drop(['log_Amount', 'Amount'], axis=1)
y_total = combined_data['log_Amount']

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
# calculate the root mean squared error and mean absolute error of the  predictions compared to the actual values
b_mse = mean_squared_error(df_test_y, b_y_predict)
b_mae = mean_absolute_error(df_test_y, b_y_predict)
# b_mse = mean_squared_error(y, b_y_predict)
# b_mae = mean_absolute_error(y, b_y_predict)
print("Backward selection RMSE:", np.sqrt(b_mse))
print("Backward selection MAE:", b_mae)

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


# Define the number of folds for k-fold cross-validation
k_folds = 5
kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

# Initialize variables to store results
backward_rmse_list = []
backward_mae_list = []
forward_rmse_list = []
forward_mae_list = []

# Iterate over the folds
for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Backward Selection
    backselect_fold = step_reg.backward_regression(
        X_train, y_train, 0.05, verbose=False)
    X_backselect_fold = X_train[backselect_fold]
    back_model_fold = sm.OLS(y_train, X_backselect_fold)
    fit_back_model_fold = back_model_fold.fit()
    b_y_predict_fold = fit_back_model_fold.predict(X_backselect_fold)
    b_mse_fold = mean_squared_error(y_train, b_y_predict_fold)
    b_mae_fold = mean_absolute_error(y_train, b_y_predict_fold)
    backward_rmse_list.append(np.sqrt(b_mse_fold))
    backward_mae_list.append(b_mae_fold)

    # Forward Selection
    forwardselect_fold = step_reg.forward_regression(
        X_train, y_train, 0.05, verbose=False)
    X_forwardselect_fold = X_train[forwardselect_fold]
    forward_model_fold = sm.OLS(y_train, X_forwardselect_fold)
    fit_forward_model_fold = forward_model_fold.fit()
    f_y_predict_fold = fit_forward_model_fold.predict(X_forwardselect_fold)
    f_mse_fold = mean_squared_error(y_train, f_y_predict_fold)
    f_mae_fold = mean_absolute_error(y_train, f_y_predict_fold)
    forward_rmse_list.append(np.sqrt(f_mse_fold))
    forward_mae_list.append(f_mae_fold)

# Display average results over k folds
print("\nk-Fold Cross Validation Results:")
print("Backward Selection Average RMSE:", np.mean(backward_rmse_list))
print("Backward Selection Average MAE:", np.mean(backward_mae_list))
print("Forward Selection Average RMSE:", np.mean(forward_rmse_list))
print("Forward Selection Average MAE:", np.mean(forward_mae_list))

# make predictions using the new model with selected features from forward selection
f_y_predict = fit_forward_model.predict(X_forwardselect)
f_y_predict = f_y_predict[:len(df_test_y)]
# calculate the root mean squared error and mean absolute error of the predictions compared to the actual values
f_mse = mean_squared_error(df_test_y, f_y_predict)
f_mae = mean_absolute_error(df_test_y, f_y_predict)
# f_mse = mean_absolute_error(y, f_y_predict)
# f_mse = mean_squared_error(y, f_y_predict)
print("Forward selection RMSE:", np.sqrt(f_mse))
print("Forward selection MAE:", f_mae)

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

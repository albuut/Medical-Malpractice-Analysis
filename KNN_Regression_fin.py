import sys 
import pandas as pd
import numpy as np
import scipy.stats as stats
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from factor_analyzer import FactorAnalyzer
from sklearn.random_projection import GaussianRandomProjection
from feature_engine.outliers import OutlierTrimmer
from scipy.spatial import KDTree
from sklearn.metrics import mean_squared_error, mean_absolute_error # used to evaluate the quality of model for comparison purposes
from math import sqrt


# define knn regression function
# Read through this documentation for KDTree(https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.KDTree.query.html)
def knnRegressor(X_train, X_test, y_train, k):
    kd_tree = KDTree(X_train)
    dd, ii = kd_tree.query(X_test, k=k)
    predictions = []
    for i in ii:
        nearest_neighbors = y_train.iloc[i]
        prediction = nearest_neighbors.mean()
        predictions.append(prediction)
    return np.array(predictions)
    
# Define function to find the best k
# Took some inspiration for this function here (https://www.analyticsvidhya.com/blog/2018/08/k-nearest-neighbor-introduction-regression-python/)
def bestK(X_train, X_test, y_train, y_test):
    k_values = np.linspace(2,20,num=19,dtype=int)
    mse = []
    for k in k_values:
        predictions = knnRegressor(X_train, X_test, y_train, k)
        mse.append(((y_test - predictions) ** 2).mean())
    best_k = k_values[np.argmin(mse)]
    return best_k
##################################################################

file_input = sys.argv[1]

train_suffix = '_train.csv'
test_suffix = '_test.csv'

#Read Data from Files
df_train = pd.read_csv(file_input + train_suffix)
df_test = pd.read_csv(file_input + test_suffix)

#Power transformer comment out -> [****]
'''
pt = PowerTransformer()
df_train_new = pt.fit_transform(df_train)
df_test_new = pt.transform(df_test)

#Create train and test sets for power transformer
X_train = pd.DataFrame(df_train_new, columns=df_train.columns).drop(columns=['Amount', 'log_Amount'])
y_train = df_train['log_Amount']
X_test = pd.DataFrame(df_test_new, columns=df_test.columns).drop(columns=['Amount', 'log_Amount'])
y_test = df_test['log_Amount']
''' 

#Outlier Trimmer
'''
ot = OutlierTrimmer(capping_method='gaussian', tail='both', fold=1.5, variables=['log_Amount'])
ot.fit(df_train)
df_train = ot.transform(df_train)
df_test = ot.transform(df_test)
'''

#Create train and test sets -> [****]
#'''
X_train = df_train.drop(columns=['Amount', 'log_Amount'])
y_train = df_train['log_Amount']
X_test = df_test.drop(columns=['Amount', 'log_Amount'])
y_test = df_test['log_Amount']
#'''

#Standardize training and test data
mean = X_train.mean()
std = X_train.std()
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

#pca
'''
pca = PCA(n_components=2)
pca.fit(X_train)
X_train = pca.transform(X_train)
X_test = pca.transform(X_test)
'''

#Factor Analysis
'''
fa = FactorAnalyzer()
fa.fit(X_train)
ev, v = fa.get_eigenvalues()
count = (ev > 1).sum()

fa = FactorAnalyzer(n_factors=count)
fa.fit(X_train)
X_train = fa.transform(X_train)
X_test = fa.transform(X_test)
'''

#Random Projection
'''
#mse 2=.1579 5=.114  10=.112 15=.1109 20=.1103 25=.1107 24=.11021
rp = GaussianRandomProjection(n_components=24,random_state=42)
rp.fit(X_train)
X_train = rp.transform(X_train)
X_test = rp.transform(X_test)
'''

# Find best k
best_k = bestK(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
# Fit K nearest neighbors and return predictions
predict = knnRegressor(X_train=X_train, X_test=X_test, y_train=y_train, k=best_k)

# Cross Validation
# The custom_cross_validate_knn function came from chatgpt
#*********************************************************************************
def custom_cross_validate_knn(X_train, y_train, X_test, y_test, cv=5):
     kf = KFold(n_splits=cv, shuffle=True, random_state=42)
     rmse_scores = []
     mae_scores = []

    # Obtain the best 'k' using the provided function
     best_k = bestK(X_train, X_test, y_train, y_test)

     for train_index, val_index in kf.split(X_train):
        X_cv_train, X_cv_val = X_train.iloc[train_index], X_train.iloc[val_index]
        y_cv_train, y_cv_val = y_train.iloc[train_index], y_train.iloc[val_index]

        # Make predictions using your knnRegressor function with the best 'k'
        y_pred = knnRegressor(X_cv_train.values, X_cv_val.values, y_cv_train, best_k)

        # Calculate evaluation metrics (RMSE and MAE)
        rmse = np.sqrt(mean_squared_error(y_cv_val, y_pred))
        mae = mean_absolute_error(y_cv_val, y_pred)

        rmse_scores.append(rmse)
        mae_scores.append(mae)

     return rmse_scores, mae_scores, best_k

custom_rmse_scores, custom_mae_scores, best_k = custom_cross_validate_knn(X_train, y_train, X_test, y_test)
#**************************************************************************************************************
# Calculate mean and standard deviation of the custom RMSE and MAE scores
mean_rmse = np.mean(custom_rmse_scores)
std_rmse = np.std(custom_rmse_scores)
mean_mae = np.mean(custom_mae_scores)
std_mae = np.std(custom_mae_scores)

# Output results for your knnRegressor and best 'k'
print("Mean Absolute Error (CV):", mean_mae)
print("Standard Deviation of MAE (CV):", std_mae)
print("Root Mean Squared Error (CV):", mean_rmse)
print("Standard Deviation of RMSE (CV):", std_rmse)
print("===================================================")

#Calculate mse, rmse, T stat, P Val
mse = mean_squared_error(y_test, predict)
mae = mean_absolute_error(y_test, predict)
rmse = sqrt(mse)
t_stat, p_val = stats.ttest_ind(y_test, predict)
print("Mean Absolute Error: ", mae)
print("Root Mean Squared Error: ", rmse)
print("T-Statistic: ", t_stat)
print("P-Value: ", p_val)

#print(predict)
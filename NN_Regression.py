import sys 
import pandas as pd
import scipy.stats as stats
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from factor_analyzer import FactorAnalyzer
from sklearn.random_projection import GaussianRandomProjection
from feature_engine.outliers import OutlierTrimmer
from sklearn.metrics import mean_squared_error, mean_absolute_error # used to evaluate the quality of model for comparison purposes
from math import sqrt


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
df_test_new = pt.fit_transform(df_test)

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
ot.transform(df_train)
ot.transform(df_test)
'''

#Create train and test sets -> [****]
#'''
X_train = df_train.drop(columns=['Amount', 'log_Amount'])
y_train = df_train['log_Amount']
X_test = df_test.drop(columns=['Amount', 'log_Amount'])
y_test = df_test['log_Amount']
#'''

#Standardize training and test data
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_train = pd.DataFrame(X_train_sc)
X_test_sc = scaler.fit_transform(X_test)
X_test = pd.DataFrame(X_test_sc)

#pca
'''
pca = PCA()
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

#Find best K using grid search
params = {'n_neighbors': [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]}
knn = KNeighborsRegressor()
model = GridSearchCV(knn, params, cv=5)
model.fit(X_train, y_train)
params = model.best_params_

#Fit K nearest neighbors
model_knn = KNeighborsRegressor(n_neighbors=params.get('n_neighbors'))
model_knn.fit(X=X_train, y=y_train)
predict = model.predict(X=X_test)

#Calculate mse, rmse, T stat, P Val
mse = mean_squared_error(y_test, predict)
rmse = sqrt(mse)
t_stat, p_val = stats.ttest_ind(y_test, predict)
print("Mean Absolute Error: ", mse)
print("Root Mean Squared Error: ", rmse)
print("T-Statistic: ", t_stat)
print("P-Value: ", p_val)

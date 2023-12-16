import sys 
import pandas as pd
import numpy as np
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from factor_analyzer import FactorAnalyzer
from sklearn.random_projection import GaussianRandomProjection
from feature_engine.outliers import OutlierTrimmer
from scipy.spatial import KDTree
from sklearn.metrics import mean_squared_error, mean_absolute_error # used to evaluate the quality of model for comparison purposes
from math import sqrt


# define knn regression class
# Read through this documentation for KDTree(https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.KDTree.query.html)
class knnRegressor:
    def __init__(self, k=5):
        self.k = k
        self.X_train = None
        self.y_train = None
        self.kd_tree = None
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        self.kd_tree = KDTree(X)
    def predict(self, X):
        dd, ii = self.kd_tree.query(X, k=self.k)
        predictions = []
        for i in ii:
            nearest_neighbors = self.y_train.iloc[i]
            prediction = nearest_neighbors.mean()
            predictions.append(prediction)
        return np.array(predictions)

# Define function to find the best k
# Took some inspiration for this function here (https://www.analyticsvidhya.com/blog/2018/08/k-nearest-neighbor-introduction-regression-python/)
def bestK(X_train, X_test, y_train, y_test):
    k_values = np.linspace(2,20,num=19,dtype=int)
    mse = []
    for k in k_values:
        model_knn = knnRegressor(k=k)
        model_knn.fit(X_train,y_train)
        predictions = model_knn.predict(X_test)
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
model_knn = knnRegressor(k=best_k)
model_knn.fit(X=X_train, y=y_train)
predict = model_knn.predict(X=X_test)


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
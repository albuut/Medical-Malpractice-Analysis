import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error # used to evaluate the quality of model for comparison purposes
import sys 

file_input = sys.argv[1]

train_suffix = '_train.csv'
validate_suffix = '_validate.csv'
test_suffix = '_test.csv'

#Read Data from Files
df_train = pd.read_csv(file_input + train_suffix)
df_validate = pd.read_csv(file_input + validate_suffix)
df_test = pd.read_csv(file_input + test_suffix)

#Create train and test sets
X_train = df_train.drop(columns=['Amount', 'log_Amount'])
y_train = df_train['log_Amount']
X_test = df_test.drop(columns=['Amount', 'log_Amount'])
y_test = df_test['log_Amount']

#Standardize training and test data
scaler = StandardScaler()

X_train_sc = scaler.fit_transform(X_train)
X_train = pd.DataFrame(X_train_sc)
X_test_sc = scaler.fit_transform(X_test)
X_test = pd.DataFrame(X_test_sc)
X_new_sc = scaler.transform(pd.DataFrame(df_validate.drop(columns=['Amount', 'log_Amount'])))

#pca

pca = PCA()
pca.fit(X_train)
X_train = pca.transform(X_train)
X_test = pca.transform(X_test)


#Factor Analysis


#Find best K using grid search
params = {'n_neighbors': [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]}
knn = KNeighborsRegressor()
model = GridSearchCV(knn, params, cv=5)
model.fit(X_train, y_train)
params = model.best_params_

#Fit K nearest neighbors
model_knn = KNeighborsRegressor(n_neighbors=params.get('n_neighbors'))
model_knn.fit(X=X_train_sc, y=y_train)
predict = model.predict(X=X_new_sc)

#Putting predictions in csv file
results = pd.read_csv(file_input + '_validate.csv')
results = results.loc[:, results.columns.intersection(['log_Amount'])]
results["model_Amount"] = predict
results.to_csv('KNN_results.csv', index =False)

#Find mean squared error
print("Mean squared error: ", mean_squared_error(results['log_Amount'].values, results['model_Amount'].values))

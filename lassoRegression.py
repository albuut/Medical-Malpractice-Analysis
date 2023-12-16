import sys
import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error

from sklearn.preprocessing import PowerTransformer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.decomposition import FastICA
from sklearn.preprocessing import PolynomialFeatures
from feature_engine.outliers import OutlierTrimmer

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from scipy import stats

from sklearn.model_selection import cross_val_predict


#default
'''

df_train = pd.read_csv('medicalmalpractice.csv_train.csv')
df_test = pd.read_csv('medicalmalpractice.csv_test.csv')

X_train = df_train.drop(columns=['Amount', 'log_Amount'])
y_train = df_train['log_Amount']

X_test = df_test.drop(columns=['Amount', 'log_Amount'])
y_test = df_test['log_Amount']

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lasso = Lasso(alpha=0.001)

y_pred_cv = cross_val_predict(lasso, X_train_scaled, y_train, cv=10)

mae_cv = mean_absolute_error(y_train, y_pred_cv)
mse_cv = mean_squared_error(y_train, y_pred_cv)

print("Cross-Validation Results:")
print(f"Mean Absolute Error (CV): {mae_cv}")
print(f"Mean Squared Error (CV): {mse_cv}")

lasso.fit(X_train_scaled, y_train)

coefficients = lasso.coef_
feature_names = X_train.columns

feature_coefficients = list(zip(feature_names, coefficients))

feature_coefficients.sort(key=lambda x: abs(x[1]), reverse=True)

print("2 Most Important Features:")
for feature, coefficient in feature_coefficients[:2]:
    print(f"{feature}: {coefficient}")

y_pred = lasso.predict(X_test_scaled)

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.1)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs. Predicted")
plt.grid(True)
plt.show()

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print("Test Results:")
print(f"Mean Absolute Error (Test): {mae}")
print(f"Mean Squared Error (Test): {mse}")

'''



#Spline
'''
df_train = pd.read_csv('medicalmalpractice.csv_train.csv')
df_test = pd.read_csv('medicalmalpractice.csv_test.csv')

X_train = df_train.drop(columns=['Amount', 'log_Amount'])
y_train = df_train['log_Amount']

X_test = df_test.drop(columns=['Amount', 'log_Amount'])
y_test = df_test['log_Amount']

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

spline_transformer = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = spline_transformer.fit_transform(X_train_scaled)
X_test_poly = spline_transformer.transform(X_test_scaled)

lasso = Lasso(alpha=0.001)
lasso.fit(X_train_poly, y_train)
coefficients = lasso.coef_
feature_names = X_train.columns

feature_coefficients = list(zip(feature_names, coefficients))

feature_coefficients.sort(key=lambda x: abs(x[1]), reverse=True)

print("2 Most Important Features:")
for feature, coefficient in feature_coefficients[:2]:
    print(f"{feature}: {coefficient}")
y_pred = lasso.predict(X_test_poly)

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.1)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs. Predicted")
plt.grid(True)
plt.show()

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
'''

#PCA
'''
df_train = pd.read_csv('medicalmalpractice.csv_train.csv')
df_test = pd.read_csv('medicalmalpractice.csv_test.csv')

X_train = df_train.drop(columns=['Amount', 'log_Amount'])
y_train = df_train['log_Amount']

X_test = df_test.drop(columns=['Amount', 'log_Amount'])
y_test = df_test['log_Amount']

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("PCA:")
n_components = 2 
pca = PCA(n_components=n_components)
X_train_poly = pca.fit_transform(X_train_scaled)
X_test_poly = pca.transform(X_test_scaled)

lasso = Lasso(alpha=0.001)
lasso.fit(X_train_poly, y_train)
coefficients = lasso.coef_
feature_names = X_train.columns

feature_coefficients = list(zip(feature_names, coefficients))

feature_coefficients.sort(key=lambda x: abs(x[1]), reverse=True)

print("2 Most Important Features:")
for feature, coefficient in feature_coefficients[:2]:
    print(f"{feature}: {coefficient}")
y_pred = lasso.predict(X_test_poly)

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.1)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs. Predicted")
plt.grid(True)
plt.show()

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
'''

#Power Transformer
'''
df_train = pd.read_csv('medicalmalpractice.csv_train.csv')
df_test = pd.read_csv('medicalmalpractice.csv_test.csv')

X_train = df_train.drop(columns=['Amount', 'log_Amount'])
y_train = df_train['log_Amount']
X_test = df_test.drop(columns=['Amount', 'log_Amount'])
y_test = df_test['log_Amount']

pt = PowerTransformer()
X_train_power = pt.fit_transform(X_train)
X_test_power = pt.transform(X_test)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_power)
X_test_scaled = scaler.transform(X_test_power)

lasso = Lasso(alpha=0.01)
lasso.fit(X_train_scaled, y_train)
coefficients = lasso.coef_
feature_names = X_train.columns

feature_coefficients = list(zip(feature_names, coefficients))

feature_coefficients.sort(key=lambda x: abs(x[1]), reverse=True)

print("2 Most Important Features:")
for feature, coefficient in feature_coefficients[:2]:
    print(f"{feature}: {coefficient}")
y_pred = lasso.predict(X_test_scaled)

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.1)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs. Predicted")
plt.grid(True)
plt.show()

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print("Test Results:")
print(f"Mean Absolute Error (Test): {mae}")
print(f"Mean Squared Error (Test): {mse}")
'''

#Outlier Trimming
'''
df_train = pd.read_csv('medicalmalpractice.csv_train.csv')
df_test = pd.read_csv('medicalmalpractice.csv_test.csv')

ot = OutlierTrimmer(capping_method='gaussian', tail='both', fold=1.5, variables=['log_Amount'])
ot.fit(df_train)
df_train = ot.transform(df_train)
df_test = ot.transform(df_test)

X_train = df_train.drop(columns=['Amount', 'log_Amount'])
y_train = df_train['log_Amount']
X_test = df_test.drop(columns=['Amount', 'log_Amount'])
y_test = df_test['log_Amount']

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lasso = Lasso(alpha=0.01)
lasso.fit(X_train_scaled, y_train)
coefficients = lasso.coef_
feature_names = X_train.columns

feature_coefficients = list(zip(feature_names, coefficients))

feature_coefficients.sort(key=lambda x: abs(x[1]), reverse=True)

print("2 Most Important Features:")
for feature, coefficient in feature_coefficients[:2]:
    print(f"{feature}: {coefficient}")
y_pred = lasso.predict(X_test_scaled)

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.1)
plt.xlabel("Actual Values")
plt.ylabel ("Predicted Values")
plt.title("Actual vs. Predicted")
plt.grid(True)
plt.show()

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print("Test Results:")
print(f"Mean Absolute Error (Test): {mae}")
print(f"Mean Squared Error (Test): {mse}")
'''

#tsne
df_train = pd.read_csv('medicalmalpractice.csv_train.csv')
df_test = pd.read_csv('medicalmalpractice.csv_test.csv')

X_train = df_train.drop(columns=['Amount', 'log_Amount'])
y_train = df_train['log_Amount']
X_test = df_test.drop(columns=['Amount', 'log_Amount'])
y_test = df_test['log_Amount']

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

n_components = 2 
tsne = TSNE(n_components=n_components, random_state=42)
X_train_tsne = tsne.fit_transform(X_train_scaled)
X_test_tsne = tsne.fit_transform(X_test_scaled)

lasso = Lasso(alpha=0.01)
lasso.fit(X_train_tsne, y_train)
coefficients = lasso.coef_
feature_names = X_train.columns

feature_coefficients = list(zip(feature_names, coefficients))

feature_coefficients.sort(key=lambda x: abs(x[1]), reverse=True)

print("2 Most Important Features:")
for feature, coefficient in feature_coefficients[:2]:
    print(f"{feature}: {coefficient}")
y_pred = lasso.predict(X_test_tsne)

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.1)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs. Predicted")
plt.grid(True)
plt.show()

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print("Test Results:")
print(f"Mean Absolute Error (Test): {mae}")
print(f"Mean Squared Error (Test): {mse}")

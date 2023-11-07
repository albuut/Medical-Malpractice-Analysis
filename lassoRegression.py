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

# Apply feature scaling using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create a Lasso regression model
lasso = Lasso(alpha=0.001)

# Perform cross-validation on the scaled data
y_pred_cv = cross_val_predict(lasso, X_train_scaled, y_train, cv=10)

# Evaluate the model
mae_cv = mean_absolute_error(y_train, y_pred_cv)
mse_cv = mean_squared_error(y_train, y_pred_cv)

print("Cross-Validation Results:")
print(f"Mean Absolute Error (CV): {mae_cv}")
print(f"Mean Squared Error (CV): {mse_cv}")

# Fit the Lasso model to the entire training data (scaled)
lasso.fit(X_train_scaled, y_train)

# Make predictions on the test data (scaled)
y_pred = lasso.predict(X_test_scaled)

# Plot actual vs. predicted values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.1)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs. Predicted")
plt.grid(True)
plt.show()

# Evaluate the model on the test data
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print("Test Results:")
print(f"Mean Absolute Error (Test): {mae}")
print(f"Mean Squared Error (Test): {mse}")



#Spline
'''
# Read the data
df_train = pd.read_csv('medicalmalpractice.csv_train.csv')
df_test = pd.read_csv('medicalmalpractice.csv_test.csv')

X_train = df_train.drop(columns=['Amount', 'log_Amount'])
y_train = df_train['log_Amount']

X_test = df_test.drop(columns=['Amount', 'log_Amount'])
y_test = df_test['log_Amount']

# Apply feature scaling using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Apply polynomial transformation (if needed)
spline_transformer = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = spline_transformer.fit_transform(X_train_scaled)
X_test_poly = spline_transformer.transform(X_test_scaled)

# Create and fit the Lasso model
lasso = Lasso(alpha=0.001)
lasso.fit(X_train_poly, y_train)

# Make predictions on the test data
y_pred = lasso.predict(X_test_poly)

# Plot actual vs. predicted values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.1)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs. Predicted")
plt.grid(True)
plt.show()

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")

'''

#PCA
'''
# Read the data
df_train = pd.read_csv('medicalmalpractice.csv_train.csv')
df_test = pd.read_csv('medicalmalpractice.csv_test.csv')

X_train = df_train.drop(columns=['Amount', 'log_Amount'])
y_train = df_train['log_Amount']

X_test = df_test.drop(columns=['Amount', 'log_Amount'])
y_test = df_test['log_Amount']

# Apply feature scaling using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#pca
print("PCA:")
n_components = 2  # Number of principal components to retain
pca = PCA(n_components=n_components)
X_train_poly = pca.fit_transform(X_train_scaled)
X_test_poly = pca.transform(X_test_scaled)

# Create and fit the Lasso model
lasso = Lasso(alpha=0.001)
lasso.fit(X_train_poly, y_train)

# Make predictions on the test data
y_pred = lasso.predict(X_test_poly)

# Plot actual vs. predicted values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.1)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs. Predicted")
plt.grid(True)
plt.show()

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
'''

#Power Transformer
'''
# Read the data
df_train = pd.read_csv('medicalmalpractice.csv_train.csv')
df_test = pd.read_csv('medicalmalpractice.csv_test.csv')

# Separate the target variable and the features
X_train = df_train.drop(columns=['Amount', 'log_Amount'])
y_train = df_train['log_Amount']
X_test = df_test.drop(columns=['Amount', 'log_Amount'])
y_test = df_test['log_Amount']

# Power transform the data
pt = PowerTransformer()
X_train_power = pt.fit_transform(X_train)
X_test_power = pt.transform(X_test)

# Apply feature scaling using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_power)
X_test_scaled = scaler.transform(X_test_power)

# Create a Lasso regression model
lasso = Lasso(alpha=0.01)
lasso.fit(X_train_scaled, y_train)

# Make predictions on the test data
y_pred = lasso.predict(X_test_scaled)

# Plot actual vs. predicted values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.1)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs. Predicted")
plt.grid(True)
plt.show()

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print("Test Results:")
print(f"Mean Absolute Error (Test): {mae}")
print(f"Mean Squared Error (Test): {mse}")
'''

#Outlier Trimming
'''
# Read the data
df_train = pd.read_csv('medicalmalpractice.csv_train.csv')
df_test = pd.read_csv('medicalmalpractice.csv_test.csv')

# Outlier trimming
ot = OutlierTrimmer(capping_method='gaussian', tail='both', fold=1.5, variables=['log_Amount'])
ot.fit(df_train)
df_train = ot.transform(df_train)
df_test = ot.transform(df_test)

# Separate the target variable and the features
X_train = df_train.drop(columns=['Amount', 'log_Amount'])
y_train = df_train['log_Amount']
X_test = df_test.drop(columns=['Amount', 'log_Amount'])
y_test = df_test['log_Amount']

# Apply feature scaling using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create a Lasso regression model
lasso = Lasso(alpha=0.01)
lasso.fit(X_train_scaled, y_train)

# Make predictions on the test data
y_pred = lasso.predict(X_test_scaled)

# Plot actual vs. predicted values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.1)
plt.xlabel("Actual Values")
plt.ylabel ("Predicted Values")
plt.title("Actual vs. Predicted")
plt.grid(True)
plt.show()

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print("Test Results:")
print(f"Mean Absolute Error (Test): {mae}")
print(f"Mean Squared Error (Test): {mse}")

'''

#tsne
'''
# Read the data
df_train = pd.read_csv('medicalmalpractice.csv_train.csv')
df_test = pd.read_csv('medicalmalpractice.csv_test.csv')

# Separate the target variable and the features
X_train = df_train.drop(columns=['Amount', 'log_Amount'])
y_train = df_train['log_Amount']
X_test = df_test.drop(columns=['Amount', 'log_Amount'])
y_test = df_test['log_Amount']

# Apply feature scaling using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Perform t-SNE on the scaled data
n_components = 2  # Number of dimensions in the lower-dimensional space
tsne = TSNE(n_components=n_components, random_state=42)
X_train_tsne = tsne.fit_transform(X_train_scaled)
X_test_tsne = tsne.fit_transform(X_test_scaled)

# Create a Lasso regression model
lasso = Lasso(alpha=0.01)
lasso.fit(X_train_tsne, y_train)

# Make predictions on the test data (t-SNE transformed)
y_pred = lasso.predict(X_test_tsne)

# Plot actual vs. predicted values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.1)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs. Predicted")
plt.grid(True)
plt.show()

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print("Test Results:")
print(f"Mean Absolute Error (Test): {mae}")
print(f"Mean Squared Error (Test): {mse}")
'''

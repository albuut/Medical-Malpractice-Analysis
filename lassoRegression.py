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

file_input = sys.argv[1]

df = pd.read_csv(file_input)

train_suffix = '_train.csv'
validate_suffix = '_validate.csv'
test_suffix = '_test.csv'

df_train = pd.read_csv(file_input + train_suffix)
df_validate = pd.read_csv(file_input + validate_suffix)
df_test = pd.read_csv(file_input + test_suffix)

X_train = df_train.drop(columns=['Amount', 'log_Amount'])
X_test = df_test.drop(columns=['Amount', 'log_Amount'])
y_train = df_train['log_Amount']
y_test = df_test['log_Amount']

# #Spline Transformer
# print("Spline Transformer:")
# spline_transformer = PolynomialFeatures(degree=3, include_bias=False)
# X_train = spline_transformer.fit_transform(X_train)
# X_test = spline_transformer.transform(X_test)

# #pca
# print("PCA:")
# n_components = 2  # Number of principal components to retain
# pca = PCA(n_components=n_components)
# X_train = pca.fit_transform(X_train)
# X_test = pca.transform(X_test)

# # Power transformer
# print("Power Transformer:")
# pt = PowerTransformer()
# X_train = pt.fit_transform(X_train)
# X_test = pt.transform(X_test)

# # Independent Component Analysis
# print("Independent Component Analysis:")
# n_components = 24  # Number of components to extract
# ica = FastICA(n_components=n_components, random_state=42)
# X_train = ica.fit_transform(X_train)
# X_test = ica.transform(X_test)

# # t-SNE
# print("t-SNE:")
# n_components = 2  # Number of dimensions in the lower-dimensional space
# tsne = TSNE(n_components=n_components, random_state=42)
# X_train = tsne.fit_transform(X_train)
# X_test = tsne.fit_transform(X_test)

lasso = Lasso()
lasso.fit(X_train, y_train)

y_pred = lasso.predict(X_test)

print("Predictions on Test Data:")
print(y_pred)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import cross_val_predict
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

import numpy as np

class LassoRegression:
    def __init__(self, alpha=0.001, max_iter=1000, tol=1e-4, learning_rate=0.01):
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.learning_rate = learning_rate
        self.coef_ = None

    def soft_threshold(self, x, threshold):
        return np.sign(x) * np.maximum(np.abs(x) - threshold, 0.0)

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.coef_ = np.zeros(n_features)
        X_transpose = X.T
        old_coef = np.copy(self.coef_)
        
        for _ in range(self.max_iter):
            # Compute predictions
            y_pred = np.dot(X, self.coef_)

            # Compute gradient
            non_zero_mask = np.sign(self.coef_) != 0
            gradient = -(2/n_samples) * np.dot(X_transpose, (y - y_pred)) + self.alpha * np.sign(self.coef_) * non_zero_mask

            # Update coefficients using soft-thresholding
            self.coef_ -= self.learning_rate * gradient

            # Check for convergence
            if np.linalg.norm(self.coef_ - old_coef, ord=np.inf) < self.tol:
                break
            old_coef = np.copy(self.coef_)

    def predict(self, X):
        return np.dot(X, self.coef_) + 5

    def get_params(self, deep=True):
        return {'alpha': self.alpha, 'max_iter': self.max_iter, 'tol': self.tol, 'learning_rate': self.learning_rate}

    def set_params(self, **params):
        for param, value in params.items():
            setattr(self, param, value)
        return self


# Load data
df_train = pd.read_csv('medicalmalpractice.csv_train.csv')
df_test = pd.read_csv('medicalmalpractice.csv_test.csv')

X_train = df_train.drop(columns=['Amount', 'log_Amount'])
y_train = df_train['log_Amount']

X_test = df_test.drop(columns=['Amount', 'log_Amount'])
y_test = df_test['log_Amount']

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and fit your custom Lasso regression model
lasso_custom = LassoRegression(alpha=0.001)
y_pred_cv = cross_val_predict(lasso_custom, X_train_scaled, y_train, cv=10)

# Calculate cross-validation metrics
mae_cv = mean_absolute_error(y_train, y_pred_cv)
mse_cv = mean_squared_error(y_train, y_pred_cv)

print("Cross-Validation Results:")
print(f"Mean Absolute Error (CV): {mae_cv}")
print(f"Mean Squared Error (CV): {mse_cv}")

# Fit the model on the full training set
lasso_custom.fit(X_train_scaled, y_train)

# Extract and print coefficients
coefficients = lasso_custom.coef_
feature_names = X_train.columns

feature_coefficients = list(zip(feature_names, coefficients))
feature_coefficients.sort(key=lambda x: abs(x[1]), reverse=True)

print("2 Most Important Features:")
for feature, coefficient in feature_coefficients[:2]:
    print(f"{feature}: {coefficient}")

# Make predictions on the test set
y_pred = lasso_custom.predict(X_test_scaled)

# Visualize actual vs. predicted values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.1)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs. Predicted")
plt.grid(True)
plt.show()

# Evaluate test set performance
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred) 

print("Test Results:")
print(f"Mean Absolute Error (Test): {mae}")
print(f"Mean Squared Error (Test): {mse}")


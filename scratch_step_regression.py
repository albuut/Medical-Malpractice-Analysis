import sys
import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, PowerTransformer
from feature_engine.outliers import OutlierTrimmer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.random_projection import GaussianRandomProjection
from sklearn.preprocessing import PolynomialFeatures


# Backward selection will iteratively remove features based on the p-value threshold
def backward_selection(X, y, threshold=0.05):
    # List of feature names
    features = list(X.columns)
    num_features = len(features)
    num_observations = len(y)

    # Iterate through the features (loop variable is not used)
    for i in range(num_features):
        # Create column for the intercept
        X_with_constant = np.column_stack((np.ones(num_observations), X))
        # Calculate weights of features in the model
        coefficients = np.linalg.lstsq(X_with_constant, y, rcond=None)[0]

        # Calculate residuals (difference between the observed and predicted values)
        residuals = y - np.dot(X_with_constant, coefficients)

        # Calculate the standard error of the coefficients
        se = np.sqrt(np.sum(residuals**2) /
                     (num_observations - num_features - 1))
        se_coefficients = se * \
            np.linalg.inv(np.dot(X_with_constant.T,
                          X_with_constant)).diagonal()
        # Handle division by zero
        se_coefficients[se_coefficients == 0] = np.inf

        # Calculate t-statistics and p-values
        t_statistics = coefficients / se_coefficients
        p_values = 2 * (1 - 0.5 * (1 + np.sign(t_statistics) *
                        np.sqrt(1 - np.exp(-2 * t_statistics**2))))
        max_p_value = p_values[1:].max()

        # If the max p-value is above the threshold, remove the feature
        if max_p_value > threshold:
            # Stores feature with largest p-value (ignoring the intercept term)
            remove_feature = features[np.argmax(p_values[1:])]
            features.remove(remove_feature)
            X = X[features]
        # All features deemed insignificant have been removed
        else:
            break

    return features


# Forward selection will iteratively add features based on the p-value threshold
def forward_selection(X, y, threshold=0.05):
    # List of feature names
    features = list(X.columns)
    num_features = len(features)
    num_observations = len(y)
    selected_features = []

    # Iterate through the features (loop variable is not used)
    for i in range(num_features):
        best_p_value = np.inf
        best_feature = None

        for feature in features:
            # Add the current feature to evaluate to selected features
            X_selected = X[selected_features + [feature]]
            # Create column for the intercept
            X_with_constant = np.column_stack(
                (np.ones(num_observations), X_selected))
            # Calculate the weights of features in the model
            coefficients = np.linalg.lstsq(X_with_constant, y, rcond=None)[0]

            # Calculate residuals (difference between the observed and predicted values)
            residuals = y - np.dot(X_with_constant, coefficients)

            # Calculate the standard error (se) of the coefficients
            se = np.sqrt(np.sum(residuals**2) /
                         (num_observations - len(selected_features) - 1))
            se_coefficients = se * \
                np.linalg.inv(np.dot(X_with_constant.T,
                              X_with_constant)).diagonal()
            # Handle division by zero (caused by potential collinearity). Set to infinity, so feature will not be added.
            se_coefficients[se_coefficients == 0] = np.inf
            # Calculate t-statistics and p-values
            t_statistics = coefficients / se_coefficients
            p_value = 2 * (1 - 0.5 * (1 + np.sign(t_statistics)
                           * np.sqrt(1 - np.exp(-2 * t_statistics**2))))

            # If the p-value for the newest added feature is better, update best_feature
            if p_value[1] < best_p_value:
                best_p_value = p_value[1]
                best_feature = feature

        # If the best p_value is below the threshold, add the feature
        if best_p_value < threshold:
            selected_features.append(best_feature)
            features.remove(best_feature)
        else:
            break

    return selected_features


def linear_regression(X, y):  # Linear regression to get coefficients (feature weights)
    X_with_constant = np.column_stack((np.ones(len(y)), X))
    coefficients = np.linalg.lstsq(X_with_constant, y, rcond=None)[0]
    return coefficients


def predict(X, coefficients):  # Function to make predictions using the linear regression model
    X_with_constant = np.column_stack((np.ones(X.shape[0]), X))
    return np.dot(X_with_constant, coefficients)


# Input preprocessed .csv malpractice data file in from the command line
file_input = sys.argv[1]

# Read in inputted file
train_suffix = '_train.csv'
validate_suffix = '_validate.csv'
test_suffix = '_test.csv'

df = pd.read_csv(file_input + train_suffix)
df_test = pd.read_csv(file_input + test_suffix)

'''
# Outlier Trimmer
ot = OutlierTrimmer(capping_method='gaussian', tail='both',
                    fold=1.5, variables=['log_Amount'])
ot.fit(df)
df = ot.transform(df)
df_test = ot.transform(df_test)

# Backward Selection Metrics:
# Root Mean Squared Error: 0.29352929071517936
# Mean Absolute Error: 0.2371539729510591
# T-Statistic: -0.19536158153338734
# P-Value: 0.8451123178857909

# Forward Selection Metrics:
# Root Mean Squared Error: 0.2899315256103601
# Mean Absolute Error: 0.23436765915024999
# T-Statistic: -0.22923429327980513
# P-Value: 0.8186896868788887
'''

# Store features and target variable
X = df.drop(['log_Amount', 'Amount'], axis=1)
y = df['log_Amount']

# Backward selection testing
back_selected_features = backward_selection(X.copy(), y)
print("Features selected from backward elimination:", back_selected_features)

# Create a new data frame using the selected features from backward selection
X_train_backward = df[back_selected_features]
X_test_backward = df_test[back_selected_features]

'''
# Power Transformer
scaler_backward = StandardScaler()
pt_backward = PowerTransformer()

X_train_backward = pt_backward.fit_transform(
    scaler_backward.fit_transform(X_train_backward))
X_test_backward = pt_backward.transform(
    scaler_backward.transform(X_test_backward))

# Backward Selection Metrics:
# Root Mean Squared Error: 0.3960538810215383
# Mean Absolute Error: 0.315541649768746
# T-Statistic: 0.5397020282412468
# P-Value: 0.5894086466504025
'''

'''
# PCA
scaler_backward = StandardScaler()
pca_backward = PCA(n_components=2)

X_train_backward = pca_backward.fit_transform(
    scaler_backward.fit_transform(X_train_backward))
X_test_backward = pca_backward.transform(
    scaler_backward.transform(X_test_backward))

# Backward Selection Metrics:
# Root Mean Squared Error: 0.4337454567389689
# Mean Absolute Error: 0.3454785962826903
# T-Statistic: 0.7452119116617325
# P-Value: 0.45615252161101894
'''

'''
# Random Projection
scaler_backward = StandardScaler()
rp_backward = GaussianRandomProjection(n_components=10, random_state=42)

rp_backward.fit(X_train_backward)
X_train_backward = rp_backward.fit_transform(
    scaler_backward.fit_transform(X_train_backward))
X_test_backward = rp_backward.transform(
    scaler_backward.transform(X_test_backward))

# Backward Selection Metrics:
# Root Mean Squared Error: 0.43075964114722676
# Mean Absolute Error: 0.336636740762581
# T-Statistic: 1.144370323086886
# P-Value: 0.25248400690469
'''
'''
# Polynomial Spline Transformer
scaler_backward = StandardScaler()
spline_backward = PolynomialFeatures(degree=2, include_bias=False)

X_train_backward = spline_backward.fit_transform(
    scaler_backward.fit_transform(X_train_backward))
X_test_backward = spline_backward.transform(
    scaler_backward.transform(X_test_backward))

# Backward Selection Metrics:
# Root Mean Squared Error: 0.36316046950655695
# Mean Absolute Error: 0.2873368987172316
# T-Statistic: 0.2781822423493186
# P-Value: 0.780875387795647
'''

# Train the model
coefficients_backward = linear_regression(
    X_train_backward, df['log_Amount'])

# Make predictions on the test set for backward selection
y_pred_test_backward = predict(X_test_backward, coefficients_backward)

# Evaluate metrics for backward selection
mse_test_backward = mean_squared_error(
    df_test['log_Amount'], y_pred_test_backward)
mae_test_backward = mean_absolute_error(
    df_test['log_Amount'], y_pred_test_backward)
t_stat_backward, p_val_backward = stats.ttest_ind(
    df_test['log_Amount'], y_pred_test_backward)

print("\nBackward Selection Metrics:")
print("Root Mean Squared Error:", np.sqrt(
    mse_test_backward))  # 0.39709943647678114
print("Mean Absolute Error:", mae_test_backward)  # 0.31643322939296303
print("T-Statistic:", t_stat_backward)  # 0.5615286668588928
print("P-Value", p_val_backward)  # 0.5744435869932845


# Forward selection testing
forward_selected_features = forward_selection(X.copy(), y)
print("\n\nFeatures selected from forward selection:", forward_selected_features)

# Create a new data frame using the selected features from forward selection
X_train_forward = df[forward_selected_features]
X_test_forward = df_test[forward_selected_features]

'''
# Power Transformer
scaler_forward = StandardScaler()
pt_forward = PowerTransformer()

X_train_forward = pt_forward.fit_transform(
    scaler_forward.fit_transform(X_train_forward))
X_test_forward = pt_forward.transform(
    scaler_forward.transform(X_test_forward))

# Forward Selection Metrics:
# Root Mean Squared Error: 0.3888123152304183
# Mean Absolute Error: 0.30821124450875603
# T-Statistic: 0.5721481247130135
# P-Value: 0.5672281627775031
'''

'''
# PCA
scaler_forward = StandardScaler()
pca_forward = PCA(n_components=2)

X_train_forward = pca_forward.fit_transform(
    scaler_forward.fit_transform(X_train_forward))
X_test_forward = pca_forward.transform(
    scaler_forward.transform(X_test_forward))

# Forward Selection Metrics:
# Root Mean Squared Error: 0.4153849998079175
# Mean Absolute Error: 0.33164110332046043
# T-Statistic: 0.6321654000906143
# P-Value: 0.5272861466921093
'''

'''
# Random Projection
scaler_forward = StandardScaler()
rp_forward = GaussianRandomProjection(n_components=10, random_state=42)

rp_forward.fit(X_train_forward)
X_train_forward = rp_forward.fit_transform(
    scaler_forward.fit_transform(X_train_forward))
X_test_forward = rp_forward.transform(
    scaler_forward.transform(X_test_forward))

# Forward Selection Metrics:
# Root Mean Squared Error: 0.4308594678148278
# Mean Absolute Error: 0.3364893211702865
# T-Statistic: 0.4308594678148278
# P-Value: 0.3364893211702865
'''

'''
# Polynomial Spline Transformer
scaler_forward = StandardScaler()
spline_forward = PolynomialFeatures(degree=2, include_bias=False)

X_train_forward = spline_forward.fit_transform(
    scaler_forward.fit_transform(X_train_forward))
X_test_forward = spline_forward.transform(
    scaler_forward.transform(X_test_forward))

# Forward Selection Metrics:
# Root Mean Squared Error: 0.3550917144946819
# Mean Absolute Error: 0.2805816432954388
# T-Statistic: 0.4174581025680855
# P-Value: 0.6763479172801476
'''

# Train the model for forward selection
coefficients_forward = linear_regression(
    X_train_forward, df['log_Amount'])

# Make predictions on the test set for forward selection
y_pred_test_forward = predict(X_test_forward, coefficients_forward)

# Evaluate metrics for forward selection
mse_test_forward = mean_squared_error(
    df_test['log_Amount'], y_pred_test_forward)
mae_test_forward = mean_absolute_error(
    df_test['log_Amount'], y_pred_test_forward)
t_stat_forward, p_val_forward = stats.ttest_ind(
    df_test['log_Amount'], y_pred_test_forward)

print("\nForward Selection Metrics:")
print("Root Mean Squared Error:", np.sqrt(
    mse_test_forward))  # 0.38986819406649675
print("Mean Absolute Error:", mae_test_forward)  # 0.3090991406848745
print("T-Statistic:", t_stat_forward)  # 0.593074356643158
print("P-Value:", p_val_forward)  # 0.553138205510898

# Scatter plot for backward selection
plt.figure(figsize=(10, 6))
plt.scatter(df_test['log_Amount'], y_pred_test_backward,
            label='Backward Selection', alpha=0.7)
plt.xlabel('Actual log_Amount')
plt.ylabel('Predicted log_Amount')
plt.title('Scatter Plot for Backward Selection Model Predictions on Test Set')
plt.legend()
plt.show()

# Scatter plot for forward selection
plt.figure(figsize=(10, 6))
plt.scatter(df_test['log_Amount'], y_pred_test_forward,
            label='Forward Selection', alpha=0.7)
plt.xlabel('Actual log_Amount')
plt.ylabel('Predicted log_Amount')
plt.title('Scatter Plot for Forward Selection Model Predictions on Test Set')
plt.legend()
plt.show()

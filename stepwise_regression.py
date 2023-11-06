import sys
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error

# Input file through command line
file_input = sys.argv[1]

# Preprocessed data file
df = pd.read_csv(file_input)

# Target variable
y = df['log_Amount']

# Include all features except target and Amount
x = df.drop(['log_Amount', 'Amount'], axis=1)
x = sm.add_constant(x)
model = sm.OLS(y, x).fit()

# Backward selection
stepwise_model = model
selected_features = list(x.columns)

while True:
    p_values = stepwise_model.pvalues[1:]
    worst_feature = p_values.idxmax()
    # Features >0.05 are statistically insignificant
    if p_values[worst_feature] > 0.05:
        selected_features.remove(worst_feature)
        x_selected = x[selected_features]
        stepwise_model = sm.OLS(y, x_selected).fit()
    else:
        break

# Summary of regression
# print(stepwise_model.summary())

y_predict = stepwise_model.predict(x_selected)
mse = mean_squared_error(y, y_predict)
print("Mean Squared Error:", mse)

from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

log10_X_train = pd.read_csv('log10_data_train.csv')
log10_X_test = pd.read_csv('log10_data_test.csv')
log10_Y_train = log10_X_train['Amount']
log10_Y_test = log10_X_test['Amount']

ln_X_train = pd.read_csv('ln_data_train.csv')
ln_X_test = pd.read_csv('ln_data_test.csv')
ln_Y_train = ln_X_train['Amount']
ln_Y_test = ln_X_test['Amount']

log2_X_train = pd.read_csv('log2_data_train.csv')
log2_X_test = pd.read_csv('log2_data_test.csv')
log2_Y_train = log2_X_train['Amount']
log2_Y_test = log2_X_test['Amount']

log10_model = DecisionTreeRegressor(random_state=1)
log10_model.fit(log10_X_train, log10_Y_train)

ln_model = DecisionTreeRegressor(random_state=1)
ln_model.fit(ln_X_train, ln_Y_train)

log2_model = DecisionTreeRegressor(random_state=1)
log2_model.fit(log2_X_train, log2_Y_train)

log10_pred = log10_model.predict(log10_X_test)
ln_pred = ln_model.predict(ln_X_test)
log2_pred = log2_model.predict(log2_X_test)

log10_rmse = np.sqrt(mean_squared_error(log10_Y_test, log10_pred))
ln_rmse = np.sqrt(mean_squared_error(ln_Y_test, ln_pred))
log2_rmse = mean_squared_error(log2_Y_test, log2_pred)
log10_mae = mean_absolute_error(log10_Y_test, log10_pred)
ln_mae = mean_absolute_error(ln_Y_test, ln_pred)
log2_mae = mean_absolute_error(log2_Y_test, log2_pred)

fig, ax = plt.subplots()

ax.bar(['RMSE-log10','RMSE-ln', 'RMSE-log2', 'MAE-log10','MAE-ln', 'MAE-log2'], [log10_rmse, ln_rmse, log2_rmse, log10_mae, ln_mae, log2_mae])
ax.set_ylabel('Error')
ax.set_title('Error for Decision Tree Regression')

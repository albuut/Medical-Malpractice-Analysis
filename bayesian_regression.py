import sys
import pandas as pd
import numpy as np
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_squared_error

file_input = sys.argv[1]

#65 / 30 / 5

#stratified selection around target variable

df = pd.read_csv(file_input)
regressor = BayesianRidge()

df_train = df.iloc[:len(df)//2]
df_train_x = df_train.drop(columns=['Amount','log_Amount'])
df_train_y = df_train['log_Amount']

regressor.fit(df_train_x,df_train_y)

df_test = df.iloc[len(df)//2:]
df_test_x = df_test.drop(columns=['Amount','log_Amount'])
df_test_y = df_test['log_Amount']

df_y_predict = regressor.predict(df_test_x)
mse = mean_squared_error(df_test_y, df_y_predict)

print(mse)
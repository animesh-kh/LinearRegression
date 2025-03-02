import pandas as pd
from preprocessing import OneHotEncoder, StandardScaler
from linearregression import LinearRegressor
import numpy as np
from model_selection import test_train_split
import matplotlib.pyplot as plt

inputs = pd.read_csv('sample_employee_data.csv')
X = inputs.iloc[:, 1:-1].values
y = inputs.iloc[:, -1].values
encoder = OneHotEncoder()
X = encoder.fit_transform(X,1)
X = encoder.fit_transform(X,1)
X_train,y_train,X_test, y_test = test_train_split(X,y,0.2)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
y_train = y_train.reshape(-1,1)
Y_train = scaler.fit_transform(y_train)
regressor = LinearRegressor()
regressor.fit(X_train,y_train)
X_test = scaler.transform(X_test)
y_pred = regressor.predict(X_test)
print(y_pred)
y_test=y_test.reshape(-1,1)
print(scaler.fit_transform(y_test))



import pandas as pd
from linearregression import LinearRegressor
from sklearn.preprocessing import StandardScaler

inputs = pd.read_csv('linear_regression_data.csv')
X = inputs.iloc[:, :-1].values
y = inputs.iloc[:, -1].values
scaler = StandardScaler()
X = scaler.fit_transform(X)
regressor = LinearRegressor()
regressor.fit(X, y)

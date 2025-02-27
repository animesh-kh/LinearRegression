import pandas as pd
from preprocessing import OneHotEncoder
from linearregression import LinearRegressor

import numpy as np

inputs = pd.read_csv('sample_employee_data.csv')
X = inputs.iloc[:, 1:-1].values
y = inputs.iloc[:, -1].values
encoder = OneHotEncoder()

X = encoder.fit_transform(X,1)
X = encoder.fit_transform(X,1)
print(X)


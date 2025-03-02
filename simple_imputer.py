import numpy as np

class SimpleImputer:
    def __init__(self,missing_values=np.nan,strategy='mean'):
        self.missing_values = missing_values
        self.strategy = strategy

    def mean_imputer(self,X):
        X.fillna(X.mean(),inplace=True)
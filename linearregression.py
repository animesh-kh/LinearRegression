import numpy as np

class LinearRegressor:
    def __init__(self):
        self.Eq = None
    def fit(self,X, y):
        X = np.c_[np.ones(X.shape[0]), X]
        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.float64)
        X_transpose = np.transpose(X)
        XX_t = np.dot(X_transpose, X)
        XX_t_inverse = np.linalg.inv(XX_t)
        XX_t_inverse_X_T = np.dot(XX_t_inverse, X_transpose)
        self.Eq = np.dot(XX_t_inverse_X_T, y)
    def predict(self, X):
        X = np.c_[np.ones(X.shape[0]), X]
        return np.dot(X, self.Eq)

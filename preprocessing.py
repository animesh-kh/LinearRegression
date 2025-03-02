import numpy as np

class StandardScaler:
    def __init__(self):
        self.min = []
        self.max = []
        pass
    def fit(self, X):

        for i in range(X.shape[1]):
            self.min.append(np.min(X[:,i]))
            self.max.append(np.max(X[:,i]))
    def transform(self, X):
        for i in range(X.shape[1]):
            for j in range(X.shape[0]):
                X[j,i] = (X[j,i] - self.min[i]) / (self.max[i] - self.min[i])
        return X
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

class OneHotEncoder:
    def __init__(self):
        self.categories = []
    def fit(self, X , index):
        self.categories = list(np.unique(X[:,index]))
    def transform(self, X , index):
        one_hot_encoded = np.zeros((X.shape[0],len(self.categories)))
        for i,c in enumerate(X[:,index]):
            ind = self.categories.index(c)
            one_hot_encoded[i,ind] = 1.0
        X = np.delete(X,index,1)
        X = np.hstack((X,one_hot_encoded))
        X = np.delete(X,-1,1)
        return X
    def fit_transform(self, X , index):
        self.fit(X , index)
        return self.transform(X , index)

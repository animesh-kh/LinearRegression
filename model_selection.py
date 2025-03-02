def test_train_split(X, y,test_size):
    train_size = int(X.shape[0]*(1-test_size))
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_test = X[train_size:]
    y_test = y[train_size:]
    return X_train, y_train, X_test, y_test


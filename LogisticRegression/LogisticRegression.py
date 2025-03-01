import numpy as np 


def sigmoid(n):
    return 1/(1+np.exp(-n))
class LogisticRegression:

    def __init__(self,lr = 0.001,itr=1000):
        self.lr = lr
        self.itr = itr
        self.weights = None
        self.bias = None

    def fit(self,X,Y):
        n_samples,n_features = np.shape(X)
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.itr):
            z = np.dot(X,self.weights) + self.bias
            y_pred = sigmoid(z)
            dw = (1/n_samples) * np.dot(X.T,y_pred-Y)
            db = (1/n_samples) * np.sum(y_pred-Y)
            self.weights -= self.lr*dw
            self.bias -= self.lr*db

    def predict(self,X):
        z = np.dot(X,self.weights) + self.bias
        y_pred = sigmoid(z)
        class_pred = [0 if y<=0.5 else 1 for y in y_pred]
        return class_pred
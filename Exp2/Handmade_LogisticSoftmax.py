import numpy as np
import collections
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


class Softmax:
    def __init__(self, lr=0.001, max_iter=1000, tol=1e-4, lam=0.0001):
        self.K = 0
        self.lr = lr
        self.max_iter = max_iter
        self.tol = tol
        self.lam = lam
        self.W = 0

    def _get_K(self, y_train):
        return len(collections.Counter(y_train))

    def fit(self, X_train, y_train):
        self.K = self._get_K(y_train)
        n = len(X_train[0])
        self.W = np.zeros((self.K, n))
        w_copy = self.W.copy()

        for t in range(self.max_iter):
            for k in range(self.K):
                l = [1 if item == k else 0 for item in y_train]
                p = self.predict_proba(self.W, X_train, k)
                g = np.sum((l - p) * X_train.T, axis=1)
                self.W[k] = self.W[k] - (-self.lr * g + self.lam * self.W[k])

            if abs(np.sum(w_copy) - np.sum(self.W)) < self.tol:
                break
            else:
                w_copy = self.W.copy()

        print("Fit completed")
        return self.W

    def predict_proba(self, W, X, k):
        numerator = np.exp(np.dot(W[k], X.T))
        denominator = np.sum(np.exp(np.dot(W, X.T)), axis=0)

        return numerator / denominator

    def predict(self, X_test):
        n = len(X_test)
        prob = np.zeros((self.K, n))
        for k in range(self.K):
            prob[k] = self.predict_proba(self.W, X_test, k)

        res = np.argmax(prob, axis=0)

        return res


if __name__ == '__main__':
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123456)

    model = Softmax()
    model.fit(X_train, y_train)
    y_predict = model.predict(X_test)
    result = classification_report(y_test, y_predict)
    print(result)
    print(y_predict)

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import numpy as np


def handmade_leave_out(X, y, rate):
    length_data = len(X)
    length_training = round(rate * length_data)
    return X[0:length_training], X[length_training:-1], y[0:length_training], y[length_training:-1]


def sklearn_leave_out(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123456)
    return X_train, X_test, y_train, y_test


def handmade_k_fold(X, y, k):
    subsequent_length = round(len(X) / k)
    X_train, X_test, y_train, y_test = [], [], [], []
    for i in range(subsequent_length):
        X_test_tmp, y_test_tmp = X[i * subsequent_length:(i + 1) * subsequent_length], y[i * subsequent_length:(
                                                                                                                           i + 1) * subsequent_length]
        X_train_tmp, y_train_tmp = np.setdiff1d(X, X_test_tmp), np.setdiff1d(y, y_test_tmp)
        X_train.append(X_train_tmp)
        X_test.append(X_test_tmp)
        y_train.append(y_train_tmp)
        y_test.append(y_test_tmp)

    return X_train, X_test, y_train, y_test


def sklearn_k_fold(X, y):
    kfold = KFold(n_splits=10)
    X_train, X_test, y_train, y_test = [], [], [], []
    for train_index, test_index in kfold.split(X):
        X_train_tmp, y_train_tmp = X[train_index], y[train_index]
        X_test_tmp, y_test_tmp = X[test_index], y[test_index]
        X_train.append(X_train_tmp)
        X_test.append(X_test_tmp)
        y_train.append(y_train_tmp)
        y_test.append(y_test_tmp)

    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    iris = load_iris()
    X = iris.data
    y = iris.target
    print(type(X))

    X_train, X_test, y_train, y_test = handmade_leave_out(X, y, 0.8)
    print("Handmade Leave out:\n")
    print("X_train:\n" + str(X_train) + "\n" + "X_test:\n" + str(X_test) + "\n" + "y_train:\n" + str(
        y_train) + "\n" + "y_test:\n" + str(y_test))
    print("-" * 50)

    X_train, X_test, y_train, y_test = sklearn_leave_out(X, y)
    print("Sklearn Leave out:\n")
    print("X_train:\n" + str(X_train) + "\n" + "X_test:\n" + str(X_test) + "\n" + "y_train:\n" + str(
        y_train) + "\n" + "y_test:\n" + str(y_test))
    print("-" * 50)

    X_train, X_test, y_train, y_test = handmade_k_fold(X, y, 10)
    print("Handmade K Fold:\n")
    print("X_train:\n" + str(X_train) + "\n" + "X_test:\n" + str(X_test) + "\n" + "y_train:\n" + str(
        y_train) + "\n" + "y_test:\n" + str(y_test))
    print("-" * 50)

    X_train, X_test, y_train, y_test = sklearn_k_fold(X, y)
    print("Sklearn K Fold:\n")
    print("X_train:\n" + str(X_train) + "\n" + "X_test:\n" + str(X_test) + "\n" + "y_train:\n" + str(
        y_train) + "\n" + "y_test:\n" + str(y_test))
    print("-" * 50)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def shuffle_data(origin_data):
    """
    The method to make sure the data is randomly splited.

    Args:
        origin_data ([pd.dataframe]): Data in its original order

    Returns:
        random splited data
    """
    return origin_data.sample(frac=1)


def train_test_split(X, y, rate):
    """
    The method to split training data and testing data

    Args:
        X ([pd.dataframe]): Data without labels
        y ([pd.dataframe]): lables of the data
        rate ([float]): Size of training data

    Returns:
        training data, testing data
    """
    length_data = len(data)
    length_training = round(rate * length_data)
    return X[0:length_training], X[length_training:-1], y[0:length_training], y[length_training:-1]


def sigmoid(x):
    """
    The method to project values into 2 dimension of classification.

    Args:
        x ([array]): Input data

    Returns:
        sigmoid value
    """
    s = 1 / (1+np.exp(-x))
    return s


def gradient(X, y, beta):
    """
    The method to calculate gradient

    Args:
        X ([array]): data
        y ([array]): labels
        beta ([array]): beta

    Returns:
        gradient
    """
    X_hat = np.c_[X, np.ones((X.shape[0], 1))]
    beta = beta.reshape(-1, 1)
    y = y.reshape(-1, 1)
    p1 = sigmoid(np.dot(X_hat, beta))

    gra = (-X_hat * (y - p1)).sum(0)

    return gra.reshape(-1, 1)


def update_parameters_gradDesc(X, y, beta, learning_rate, num_iterations):
    """
    The method used to spread gradient and optimize beta.

    Args:
        X ([array]): X
        y ([array]): y
        beta ([array]): beta
        learning_rate ([float]): learning_rate
        num_iterations ([int]): num_iterations

    Returns:
        beta
    """
    for i in range(num_iterations):

        grad = gradient(X, y, beta)
        beta = beta - learning_rate * grad

        if (i % 1000 == 0):
            print('{}th iteration'.format(i))

    return beta


def predict(beta, data):
    """
    The method to predict the outcome using trained model.

    Args:
        beta ([array]): beta
        data ([array]): data

    Returns:
        the prediction
    """
    data_hat = np.c_[data, np.ones((data.shape[0], 1))]
    pro = sigmoid(np.dot(data_hat, beta))

    result = []
    for i in pro:
        if i >= 0.5:
            result.append(1)
        else:
            result.append(0)

    return result


if __name__ == "__main__":
    data = pd.read_csv('D:/Desktop/MachineLearning_DataMining_Course/Data/MelonData_191650126_qianzehao.csv')
    data = shuffle_data(data)
    y = data.pop('label')
    X = data
    # X_train, X_test, y_train, y_test = train_test_split(X, y, 0.6)
    beta = np.ones((data.shape[1]+1, 1))
    beta = update_parameters_gradDesc(
        X.values, y.values, beta, 0.01, 10000)
    print("beta:")
    print(beta)
    print("--------------------------------")
    # x_beta = np.arange(0, 1, 0.1)
    # y_beta = [-(beta[0]/beta[1])*i-(beta[2]/beta[1]) for i in x_beta]
    # plt.scatter(X_train['density'], X_train['sugar'])
    # plt.plot(x_beta, y_beta)
    # plt.show()
    prediction =predict(beta, X)
    print('Prediction is:')
    print(prediction)
    print("--------------------------------")
    acc = 0
    for i in zip(prediction,y.values):
        acc += abs(i[0]-i[1])
    acc = 1-(acc/len(prediction))
    print("Accuracy is :")
    print(acc)

from Exercise1 import Net
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs


def generate_data():
    np.random.seed(13)
    X, y_true = make_blobs(centers=4, n_samples=5000)
    return X, y_true


if __name__ == '__main__':
    X, y_true = generate_data()

    train_X = []
    train_Y = []

    iterations = 100
    train_Xs = np.vstack(X)
    train_Ys = np.vstack(y_true)

    accumulated_net = Net(input_num=2, output_num=1, hidden_num=100)
    train_loss = []

    for it in range(iterations):
        accumulated_net.forward(train_Xs)
        accumulated_net.bp(train_Ys)
        loss = accumulated_net.loss(train_Xs, train_Ys)
        train_loss.append(loss)

    line2, = plt.plot(range(iterations), train_loss)
    plt.legend([line2], ['Loss'])
    plt.show()
    print(train_loss)

from Exercise1 import Net
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

if __name__ == '__main__':
    train_X = []
    train_Y = []
    iris = load_iris()
    X = iris['data']
    Y = iris['target']

    iterations = 1000
    train_Xs = np.vstack(X)
    train_Ys = np.vstack(Y)

    standard_net = Net()
    train_loss = []

    accumulated_net = Net(input_num=4, output_num=3)
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

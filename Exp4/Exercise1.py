import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def sigmoid(x):
    f = .5 * (1 + np.tanh(.5 * x))
    return f


class Net:
    def __init__(self, hidden_num=4, output_num=1, input_num=2):
        self.hidden_num = hidden_num
        self.output_num = output_num
        self.w1 = np.random.rand(input_num, self.hidden_num)
        self.b1 = np.zeros((1, self.hidden_num))
        self.w2 = np.random.rand(self.hidden_num, self.output_num)
        self.b2 = np.zeros((1, self.output_num))
        self.y = None
        self.z2 = None
        self.h = None
        self.z1 = None
        self.X = None
        self.grad_b1 = None
        self.grad_w1 = None
        self.grad_b2 = None
        self.grad_w2 = None

    def forward(self, X):
        self.X = X
        self.z1 = X.dot(self.w1) + self.b1
        self.h = sigmoid(self.z1)
        self.z2 = self.h.dot(self.w2) + self.b2
        self.y = sigmoid(self.z2)

    def grad(self, Y):
        grad_z2 = self.y - Y
        self.grad_w2 = self.h.T.dot(grad_z2)
        self.grad_b2 = np.sum(grad_z2, axis=0, keepdims=True)
        grad_h = grad_z2.dot(self.w2.T)
        grad_z1 = grad_h * self.h * (1 - self.h)
        self.grad_w1 = self.X.T.dot(grad_z1)
        self.grad_b1 = np.sum(grad_z1, axis=0, keepdims=True)

    def update(self, lr=0.1):
        self.w1 -= lr * self.grad_w1
        self.b1 -= lr * self.grad_b1
        self.w2 -= lr * self.grad_w2
        self.b2 -= lr * self.grad_b2

    def bp(self, Y):
        self.grad(Y)
        self.update()

    def loss(self, X, Y):
        self.forward(X)
        cost = np.sum(np.log(self.y) * Y)
        return cost


if __name__ == '__main__':
    train_X = []
    train_Y = []
    data = pd.read_csv('Data/Watermelon.csv')
    X1 = data.values[:, 0]
    X2 = data.values[:, 1]
    y = data.values[:, 2]

    N = len(X1)
    for i in range(N):
        train_X.append(np.array([X1[i], X2[i]]))
        train_Y.append(y[i])

    iterations = 5000
    train_Xs = np.vstack(train_X)
    train_Ys = np.vstack(train_Y)

    standard_net = Net()
    train_loss = []
    for it in range(iterations):
        for i in range(N):
            standard_net.forward(train_X[i].reshape(1, 2))
            standard_net.bp(train_Y[i].reshape(1))

        loss = standard_net.loss(train_Xs, train_Ys)
        train_loss.append(loss)

    line1, = plt.plot(range(iterations), train_loss, "r--")

    accumulated_net = Net()
    train_loss = []
    for it in range(iterations):
        accumulated_net.forward(train_Xs)
        accumulated_net.bp(train_Ys)
        loss = accumulated_net.loss(train_Xs, train_Ys)
        train_loss.append(loss)

    line2, = plt.plot(range(iterations), train_loss, "r+")

    plt.legend([line1, line2], ['BP', 'Accumulated BP'])
    plt.show()

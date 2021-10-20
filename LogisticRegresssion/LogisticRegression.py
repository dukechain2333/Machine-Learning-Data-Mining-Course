import numpy as np


def sigmoid(x):
    s = 1 / (1+np.exp(-x))


def gradient(X, y, beta):
    X_hat = np.c_[X, np.ones((X.shape[0], 1))]
    beta = beta.reshape(-1, 1)
    y = y.reshape(-1, 1)
    p1 = sigmoid(np.dot(X_hat, beta))

    gra = (-X_hat * (y - p1)).sum(0)

    return gra.reshape(-1, 1)

def update_parameters_gradDesc(X, y, beta, learning_rate, num_iterations, print_cost):
    for i in range(num_iterations):

        grad = gradient(X, y, beta)
        beta = beta - learning_rate * grad

        if (i % 10 == 0) & print_cost:
            print('{}th iteration, cost is {}'.format(i, J_cost(X, y, beta)))

    return beta
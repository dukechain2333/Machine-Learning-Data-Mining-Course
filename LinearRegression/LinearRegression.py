import numpy as np
import matplotlib.pyplot as plt

# Size of the points dataset
m = 20

# Points x-coordinate and dummy value(x0,x1)
X0 = np.ones((m, 1))
X1 = np.arange(1, (m+1)).reshape(m, 1)
X = np.hstack((X0, X1))

# Points y-coordinate
y = np.array([3, 4, 5, 5, 2, 4, 7, 8, 11, 8, 12, 11, 13,
             13, 16, 17, 18, 17, 19, 21]).reshape(m, 1)
plt.scatter(X1, y, c='k')

# The Learning Rate lr
lr = 0.01


def error_function(theta, X, y):
    diff = np.dot(X, theta) - y
    return (1./2*m)*np.dot(np.transpose(diff), diff)


def gradient_function(theta, X, y):
    diff = np.dot(X, theta) - y
    return (1./m)*np.dot(np.transpose(X), diff)


def gradient_descent(X, y, alpha):
    theta = np.array([1, 1]).reshape(2, 1)
    gradient = gradient_function(theta, X, y)
    while not np.all(np.absolute(gradient) <= 1e-6):
        theta = theta - alpha*gradient
        gradient = gradient_function(theta, X, y)
    return theta


optimal = gradient_descent(X, y, lr)
print('optimal:', optimal)
print('error function:', error_function(optimal, X, y)[0, 0])

py = X1*optimal[1]+optimal[0]

plt.plot(X1, py, 'b')

plt.show()

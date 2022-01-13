from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import pandas as pd

if __name__ == '__main__':
    iris = load_iris()
    X = iris["data"]
    y = iris["target"]

    plt.scatter(X[:50, 2], X[:50, 3], label='setosa', marker='o')
    plt.scatter(X[50:100, 2], X[50:100, 3], label='versicolor', marker='x')
    plt.scatter(X[100:, 2], X[100:, 3], label='virginica', marker='+')
    plt.xlabel('petal length')
    plt.ylabel('petal width')
    plt.title("actual result")
    plt.legend()
    plt.show()

    kmeans = KMeans(n_clusters=3)
    kmeans.fit(X)

    center = kmeans.cluster_centers_
    num = pd.Series(kmeans.labels_).value_counts()
    y_train = pd.Series(kmeans.labels_)
    y_train.rename('res', inplace=True)
    result = pd.concat([pd.DataFrame(X), y_train], axis=1)

    Category_one = result[result['res'].values == 0]
    k1 = result.iloc[Category_one.index]
    Category_two = result[result['res'].values == 1]
    k2 = result.iloc[Category_two.index]
    Category_three = result[result['res'].values == 2]
    k3 = result.iloc[Category_three.index]

    plt.scatter(X[:50, 2], X[:50, 3], label='setosa', marker='o', c='yellow')
    plt.scatter(X[50:100, 2], X[50:100, 3], label='versicolor', marker='o', c='green')
    plt.scatter(X[100:, 2], X[100:, 3], label='virginica', marker='o', c='blue')
    plt.scatter(k1.iloc[:, 2], k1.iloc[:, 3], label='cluster_one', marker='+', c='brown')
    plt.scatter(k2.iloc[:, 2], k2.iloc[:, 3], label='cluster_two', marker='+', c='red')
    plt.scatter(k3.iloc[:, 2], k3.iloc[:, 3], label='cluster_three', marker='+', c='black')
    plt.xlabel('petal length')
    plt.ylabel('petal width')
    plt.title("result of KMeans")
    plt.legend()
    plt.show()

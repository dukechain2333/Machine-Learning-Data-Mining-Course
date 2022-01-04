# -*- coding: utf-8 -*-
"""
Created on Mon Jan  3 11:19:30 2022

@author: Administrator
"""

# In[]
"""
https://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection
"""
from sklearn import datasets
import matplotlib.pyplot as plt

iris = datasets.load_iris()
X = iris.data[:, :2]  # we only take the first two features.
y = iris.target

x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

plt.figure(2, figsize=(8, 6))
plt.clf()

# Plot the training points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1, edgecolor="k")
plt.xlabel("Sepal length")
plt.ylabel("Sepal width")

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())

# In[]
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# 使用 train/test split, random_state=4
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=2)
print(len(X_train))
print(len(X_test))

# In[]

# 评估模型的分类准确率
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
print(knn.score(X_test, y_test))

# In[]
# 改变 random_state=3
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=3)

# 评估模型的分类准确率
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
print(knn.score(X_test, y_test))

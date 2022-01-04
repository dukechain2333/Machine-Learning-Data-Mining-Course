# -*- coding: utf-8 -*-
"""
Created on Mon Jan  3 17:20:38 2022

@author: Administrator
"""
# In[]
from sklearn.datasets import load_iris

data = load_iris()
# print(dir(data))  # 查看data所具有的属性或方法
# print(data.DESCR)  # 查看数据集的简介


import pandas as pd

# 直接读到pandas的数据框中
pd.DataFrame(data=data.data, columns=data.feature_names)

import matplotlib.pyplot as plt

plt.style.use('ggplot')

X = data.data  # 只包括样本的特征，150x4
y = data.target  # 样本的类型，[0, 1, 2]
features = data.feature_names  # 4个特征的名称

targets = data.target_names  # 3类鸢尾花的名称，跟y中的3个数字对应
print(targets)
"""
将数据用图像的形式展示出来，可以对该数据集有一个直观的整体印象。
下面利用该数据集4个特征中的后两个，
即花瓣的长度和宽度，来展示所有的样本点。
"""
plt.figure(figsize=(10, 4))
plt.plot(X[:, 2][y == 0], X[:, 3][y == 0], 'bs', label=targets[0])
plt.plot(X[:, 2][y == 1], X[:, 3][y == 1], 'kx', label=targets[1])
plt.plot(X[:, 2][y == 2], X[:, 3][y == 2], 'ro', label=targets[2])
plt.xlabel(features[2])
plt.ylabel(features[3])
plt.title('Iris Data Set')
plt.legend()
plt.savefig('Iris_23.png', dpi=100)
plt.show()

# In[]
plt.figure(figsize=(10, 4))
plt.plot(X[:, 0][y == 0], X[:, 1][y == 0], 'bs', label=targets[0])
plt.plot(X[:, 0][y == 1], X[:, 1][y == 1], 'kx', label=targets[1])
plt.plot(X[:, 0][y == 2], X[:, 1][y == 2], 'ro', label=targets[2])
plt.xlabel(features[0])
plt.ylabel(features[1])
plt.title('Iris Data Set')
plt.legend()
plt.savefig('Iris_01.png', dpi=100)

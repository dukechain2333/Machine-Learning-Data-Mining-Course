import sklearn.datasets as datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

# 波士顿数据集即将下架：黑人问题

boston = datasets.load_boston()
train = boston.data
target = boston.target
X_train, X_test, y_train, y_true = train_test_split(train, target, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)
y_predict = model.predict(X_test)
result = r2_score(y_true, y_predict)
print(result)

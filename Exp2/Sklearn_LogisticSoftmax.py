from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import numpy as np
import warnings

warnings.filterwarnings("ignore")

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123456)

# Logistic Regression
model = LogisticRegression(C=2.0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
result = classification_report(y_test, y_pred)
print("The result of Logistic Regression")
print(y_pred)
print(result)

# Softmax (仅做演示)
proba = model.predict_proba(X_test)
y_pred = np.argmax(proba, axis=1)
result = classification_report(y_test, y_pred)
print("The result of Softmax")
print(y_pred)
print(result)

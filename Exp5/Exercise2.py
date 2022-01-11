import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import load_iris

if __name__ == '__main__':
    iris = load_iris()
    X = iris['data']
    y = iris['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123456)

    model = svm.SVC(kernel='linear')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print('The accuracy of Linear Core is: ' + str(accuracy_score(y_test, y_pred)))
    print(classification_report(y_test, y_pred))

from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import load_iris
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    iris = load_iris()
    X = iris["data"]
    y = iris["target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123456)

    model = AdaBoostClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    y_predict = model.predict(X_test)
    print(classification_report(y_test, y_predict))

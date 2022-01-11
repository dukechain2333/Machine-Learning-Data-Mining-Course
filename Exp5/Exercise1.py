import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

if __name__ == '__main__':
    watermelon = pd.read_csv('Data/Watermelon.csv')
    y = watermelon.pop('label')
    X = watermelon
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123456)

    # Linear Core
    model = svm.SVC(kernel='linear')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print('The accuracy of Linear Core is: ' + str(accuracy_score(y_test, y_pred)))
    print('The Support Vector of Linear Core is: ' + str(model.support_vectors_))

    # rbf
    model = svm.SVC(kernel='rbf')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print('The accuracy rbf core is: ' + str(accuracy_score(y_test, y_pred)))
    print('The Support Vector of rbf Core is: ' + str(model.support_vectors_))

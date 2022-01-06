from numpy import average
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import LeaveOneOut
import pandas as pd
import warnings

warnings.filterwarnings("ignore")


def data_preprocess(path):
    """
    The method to give columns names and split the data into features and labels

    Args:
        path ([str]): Input the path of the data

    Returns:
        [array]: X
        [array]: y
    """
    data = pd.read_csv(path, header=None)
    column_num = data.shape[1]
    header_list = ["param" + str(i) for i in range(1, column_num)]
    header_list.append("label")
    data.columns = header_list
    labels = data["label"].unique()
    for d in zip(range(len(labels)), labels):
        data.loc[data["label"] == d[1], "label"] = d[0]

    y = data.pop("label")
    X = data
    y = y.astype('int')

    return X.values, y.values


def test_with_kfold(X, y):
    """
    Split the data with KFold and train it with Logistic Regression

    Args:
        X ([array]): X
        y ([array]): y

    Returns:
        [float]: The average accuracy.
    """
    kf = KFold(n_splits=10, shuffle=True, random_state=123456)
    acc = []
    for train_index, test_index in kf.split(X):
        # print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model = LogisticRegression()
        model.fit(X_train, y_train)
        prediction = model.predict(X_test)
        acc.append(accuracy_score(y_pred=prediction, y_true=y_test))

    return average(acc)


def test_with_leave(X, y):
    """
    Split the data with Leave One out and train it with Logistic Regression

    Args:
        X ([array]): X
        y ([array]): y

    Returns:
        [float]: The average accuracy.
    """
    loo = LeaveOneOut()
    acc = []
    for train_index, test_index in loo.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model = LogisticRegression()
        model.fit(X_train, y_train)
        prediction = model.predict(X_test)
        acc.append(accuracy_score(y_pred=prediction, y_true=y_test))

    return average(acc)


if __name__ == "__main__":
    # iris dataset
    X, y = data_preprocess(r'Data/iris.data')
    acc_kfold_iris = test_with_kfold(X, y)
    print("The accuracy of classification on Iris dataset using KFold is:" +
          str(acc_kfold_iris))
    acc_leave_iris = test_with_leave(X, y)
    print("The accuracy of classification on Iris dataset using Leave One out is:" +
          str(acc_leave_iris))

    # haberman dataset
    X, y = data_preprocess(r'Data/haberman.data')
    acc_kfold_haberman = test_with_kfold(X, y)
    print("The accuracy of classification on haberman dataset using KFold is:" +
          str(acc_kfold_haberman))
    acc_leave_haberman = test_with_leave(X, y)
    print("The accuracy of classification on haberman dataset using Leave One out is:" +
          str(acc_leave_haberman))

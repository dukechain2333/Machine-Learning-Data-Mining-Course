import numpy as np
import pandas as pd


class LaNaiveByes:
    def __init__(self):
        self.train_length = None
        self.class_num = None
        self.class_value = None
        self.class_name = None
        self.class_prior = {}
        self.train_prior_ndigit = {}
        self.train_prior_digit = {}
        self.result = {}

    def fit(self, X_train, y_train):
        self.class_name = y_train.name
        self.train_length = len(y_train)
        self.class_value = y_train.value_counts()
        self.class_num = len(self.class_value)
        tmp_data = pd.concat([X_train, y_train], axis=1)
        for k in self.class_value.keys():
            self.class_prior[k] = (self.class_value[k] + 1) / (self.train_length + self.class_num)
            self.train_prior_ndigit[k] = {}
            self.train_prior_digit[k] = {}
            for col in X_train.columns:
                if X_train[col].dtype == "float64":
                    self.train_prior_digit[k][col] = {}
                    class_fit = tmp_data[[col, self.class_name]]
                    class_fit = class_fit[class_fit[self.class_name] == k]
                    self.train_prior_digit[k][col]['mean'] = class_fit[col].mean()
                    self.train_prior_digit[k][col]['var'] = class_fit[col].var()
                else:
                    class_value = X_train[col].value_counts()
                    class_num = len(class_value)
                    for ck in class_value.keys():
                        class_fit = tmp_data[(tmp_data[self.class_name] == k) & (tmp_data[col] == ck)]
                        class_fit_len = len(class_fit)
                        self.train_prior_ndigit[k][ck] = (class_fit_len + 1) / (self.class_value[k] + class_num)

    def predict(self, X_test):
        for key in self.class_value.keys():
            self.result[key] = 1
            for col in X_test.columns:
                if X_test[col].values[0] in self.train_prior_ndigit[key].keys():
                    self.result[key] *= self.train_prior_ndigit[key][X_test[col].values[0]]
                else:
                    self.result[key] *= (1 / (
                            ((2 * np.pi) ** 0.5) * (self.train_prior_digit[key][col]["var"] ** 0.5))) * np.exp(
                        -((X_test[col] - self.train_prior_digit[key][col]["mean"]) ** 2) / (
                                2 * self.train_prior_digit[key][col]["var"]))

        print("The prediction is: " + str(max(self.result)))
        return self.result


if __name__ == '__main__':
    data = pd.read_csv('Data/Watermelon.csv', index_col='编号')
    test_data = pd.read_csv('Data/TestData.csv', index_col='编号')
    y = data.pop('好瓜')
    X = data
    model = LaNaiveByes()
    model.fit(X, y)
    result = model.predict(test_data)
    print(result)

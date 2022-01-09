import pandas as pd
import math


class Node:
    def __init__(self, attribute):
        self._value = attribute
        self._child = {}

    def append_child(self, value, child):
        self._child[value] = child

    def show(self, level):
        print('This is: ' + level + '. The attribute is:' + self._value)
        for k in self._child:
            context = ''
            context += 'The route for ' + k + ' is ' + self._child[k]
            print(context)


class DecisionTree:
    def __init__(self, standard='ID3'):
        self.standard = standard
        self.tree = {}
        self.decision_index = []

    def _entropy(self, data):
        entropy = 0
        total_number = data.shape[0]

        if len(data.shape) == 1:
            class_number = data.value_counts().size
            for c in range(class_number):
                fraction = data.value_counts()[c] / total_number
                entropy += fraction * math.log(fraction, 2)

            return -entropy

        else:
            attribute_col = data.iloc[:, 0]
            attribute_index = attribute_col.value_counts().keys()
            attribute_number = [attribute_col.value_counts()[a] for a in attribute_index]
            class_col = data.iloc[:, 1]
            class_index = class_col.value_counts().keys()

            for attribute, number in zip(attribute_index, attribute_number):
                part_entropy = 0
                tmp_df = data[data.iloc[:, 0] == attribute]
                part_number = tmp_df.size / 2
                for i in class_index:
                    class_number = tmp_df[tmp_df.iloc[:, 1] == i].size / 2
                    fraction = class_number / part_number
                    try:
                        part_entropy += fraction * math.log(fraction, 2)
                    except:
                        part_entropy += 0
                entropy += (number / total_number) * (-part_entropy)

            return -entropy

    def _entropy_gain(self, total_entropy, part_entropy):
        return total_entropy + part_entropy

    def _compare(self, data):
        attribute_columns = data.columns[0:-1]
        class_column = data.columns[-1]
        total_entropy = self._entropy(data.iloc[:, -1])
        entropy_gain = []
        for i in attribute_columns:
            tmp_data = data[[i, class_column]]
            part_entropy = self._entropy(tmp_data)
            entropy_gain.append(self._entropy_gain(total_entropy, part_entropy))

        max_index = attribute_columns[entropy_gain.index(max(entropy_gain))]

        return max_index

    def fit(self, data):
        if self.standard == 'ID3':
            tmp_data = data.copy()
            level = 1
            while True:
                max_index = self._compare(tmp_data)
                self.decision_index.append(max_index)
                values = data[max_index].value_counts().keys()
                level_name = 'level ' + str(level)
                level += 1
                node = Node(max_index)
                for v in values:
                    if len(data[data[max_index] == v].iloc[:, -1].value_counts().keys()) == 1:
                        node.append_child(v, data[data[max_index] == v].iloc[:, -1].value_counts().keys()[0])
                    else:
                        tmp_data = tmp_data[tmp_data[max_index] == v]
                        tmp_data = tmp_data.drop(max_index, axis=1)
                        node.append_child(v, 'NextLevel')

                self.tree[level_name] = node

                if len(tmp_data.iloc[:, -1].value_counts().keys()) == 1:
                    break

            return self.tree
        else:
            return self.tree

    def show_tree(self):
        for level in self.tree.keys():
            self.tree[level].show(level)


if __name__ == '__main__':
    data = pd.read_csv('Data/loan.csv', index_col=0)
    model = DecisionTree()
    model.fit(data)
    model.show_tree()

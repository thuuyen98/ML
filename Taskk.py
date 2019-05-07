import pandas as pd
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

directory = '/Users/macos/Downloads/'
# file nay e loc cac cot bang tay :))
file_1 = '/Users/macos/Downloads/filted_train.csv'
file_2 = '/Users/macos/Downloads/filted_test.csv'

data_1 = pd.read_csv(file_1)
# print(data_1.mean())
data_1 =data_1.fillna(data_1.mean())
# print(data_1)
labels = data_1.iloc[:, :1].values.ravel()
features = data_1.iloc[:, 1:]
features = features.replace('male', 0)
features = features.replace('female', 1)
decision_tree = DecisionTreeClassifier()
features = features.values
decision_tree.fit(features, labels)

#Test
data_test = pd.read_csv(file_2)

data_test = data_test.fillna(data_test.mean())
features = data_test.replace('male', 0)
features = features.replace('female', 1)
features = features.values

prediction = decision_tree.predict(features)
data_test.insert(0, "Label", prediction)
print(data_test)
df = pd.DataFrame(data_test)
df.to_csv("taskk.csv")
import  sklearn
import pandas as pd
import pickle as pkl
import numpy
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score


names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']

dataset= pd.read_csv("/Users/macos/Downloads/iris.csv", names=names)

features= dataset.iloc[:,:4]
lables= dataset.iloc[:,4:]
features_train, features_valid, labels_train, labels_valid = train_test_split(features, lables, test_size = 0.20)
list= lables.Class.unique()
scaler = StandardScaler()
scaler.fit(features_train)
features_train = scaler.transform(features_train)
features_valid = scaler.transform(features_valid)
mlp = MLPClassifier(hidden_layer_sizes=(10, 7, 5), max_iter=300)
labels_train_1d = labels_train.values.ravel()
labels_valid_1d = labels_valid.values.ravel()
mlp=mlp.fit(features_train, labels_train_1d)
pkl.dump(mlp, open('weights1.pkl' , 'wb'))
weights= pkl.load( open('weights1.pkl','rb'))
predictions = weights.predict(features_valid)
accuracy = accuracy_score(labels_valid_1d, predictions)
print("Accuracy của model: {}%".format(round(accuracy*100,2)))

import sklearn
import pandas as pd
import numpy
import pickle as pkl
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

data_train= pd.read_csv("/Users/macos/Downloads/mnist_csv/mnist_train.csv")
data_test= pd.read_csv("/Users/macos/Downloads/mnist_csv/mnist_test.csv")
features_train= data_train.iloc[:,1:]
lables_train= data_train.iloc[:,:1]
features_valid=data_test.iloc[:,1:]
lables_valid=data_test.iloc[:,:1]
scaler = StandardScaler()
scaler.fit(features_train)
features_train = scaler.transform(features_train)
features_valid = scaler.transform(features_valid)
mlp = MLPClassifier(hidden_layer_sizes=(10, 7, 5), max_iter=200)
labels_train_1d = lables_train.values.ravel()
labels_valid_1d = lables_valid.values.ravel()
mlp=mlp.fit(features_train,labels_train_1d)

pkl.dump(mlp, open('weights.pkl' , 'wb'))
weights= pkl.load( open('weights.pkl','rb'))

predictions = weights.predict(features_valid)
print(predictions.shape)
print(labels_valid_1d.shape)
accuracy = accuracy_score(labels_valid_1d, predictions)
print("Accuracy của model: {}%".format(round(accuracy * 100, 2)))


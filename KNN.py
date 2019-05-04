import  sklearn
import pandas as pd
import pickle as pkl
import numpy
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']

dataset= pd.read_csv("/Users/macos/Downloads/iris.csv", names=names)

features= dataset.iloc[:,:4]
lables= dataset.iloc[:,4:]
features_train, features_valid, labels_train, labels_valid = train_test_split(features, lables, test_size = 0.20)
knn=KNeighborsClassifier(n_neighbors=6)
knn.fit(features_train, labels_train)
labels_pred= knn.predict(features_valid)
print("Accuracy: ",metrics.accuracy_score(labels_valid,labels_pred))



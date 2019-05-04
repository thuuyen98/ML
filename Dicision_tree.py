import  sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import  DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.externals.six import StringIO
from sklearn.tree import export_graphviz
from IPython.display import Image
import pydotplus
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width','lables']
feature_col= ['sepal-length', 'sepal-width', 'petal-length', 'petal-width']
dataset= pd.read_csv("/Users/macos/Downloads/iris.csv", names=names)

features= dataset.iloc[:,:4]
lables= dataset.iloc[:,4:]
features_train, features_valid, labels_train, labels_valid = train_test_split(features, lables, test_size = 0.20)
clf= DecisionTreeClassifier(criterion="entropy", max_depth=3)
clf.fit(features_train, labels_train)
predict= clf.predict(features_valid)

dot_data= StringIO()
export_graphviz(clf, out_file= dot_data,
                filled=True, rounded= True,
                special_characters=True, feature_names=feature_col, class_names=['0','1','2'])
graph= pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('diabetes.png')
Image(graph.create_png())
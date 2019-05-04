from sklearn.datasets import load_iris
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
iris=load_iris()
iris.keys()
df1=pd.DataFrame(iris.data, columns=iris.feature_names)
df2 =  pd.DataFrame(iris.target, columns=['type'])
df1.join(df2).head()
print(iris.target)
features= iris.data.T
sepal_len = features[0]
sepal_width = features[1]
petal_len = features[2]
petal_width = features[3]
plt.scatter(petal_len,petal_width, alpha=.5, s=100*features[3], c=iris.target,cmap='magma')
plt.show()
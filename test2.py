from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier

weather=['Sunny','Sunny','Overcast','Rainy','Rainy','Rainy','Overcast','Sunny','Sunny',
'Rainy','Sunny','Overcast','Overcast','Rainy']
# Second Feature
temp=['Hot','Hot','Hot','Mild','Cool','Cool','Cool','Mild','Cool','Mild','Mild','Mild','Hot','Mild']

# Label or target varible
play=['No','No','Yes','Yes','Yes','No','Yes','No','Yes','Yes','Yes','Yes','Yes','No']
le= preprocessing.LabelEncoder()
weather_encoded= le.fit_transform(weather)
temp_encoded=le.fit_transform(temp)
lable=le.fit_transform(play)
features= list(zip(weather_encoded,temp_encoded))
model=KNeighborsClassifier(n_neighbors=3)
model.fit(features,lable)
predicted= model.predict([[0,2]])
print(predicted)
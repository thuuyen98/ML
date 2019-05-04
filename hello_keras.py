from __future__ import print_function
from keras.datasets import fashion_mnist
import keras
from keras.layers import Dense, Flatten
from keras.models import Sequential



(x_train, y_train), (x_test, y_test)=fashion_mnist.load_data()
x_train= x_train/255
x_test= x_test/255
num_classes=10
y_train=keras.utils.to_categorical(y_train, num_classes)
y_test= keras.utils.to_categorical(y_test, num_classes)
model= Sequential()
model.add(Flatten(input_shape=(28,28)))
model.add(Dense(128, activation='relu'))
model.add(Dense(256, activation= 'relu'))
model.add(Dense(512, activation= 'relu'))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(lr=0.1),
              metrics=['accuracy'])




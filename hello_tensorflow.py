
import hello_keras as tf
from hello_keras import keras
from hello_keras.keras.models import Sequential
from hello_keras.keras.layers import Conv2D, MaxPool2D, Dropout, Dense, Flatten
from hello_keras.keras.datasets import mnist

def create_model(input_shape = (28, 28, 1)):
    # khai báo một mô hình theo dạng tuần tự
    model = keras.models.Sequential()
    # thêm một lớp Conv2D là lớp tích chập dùng cho ảnh 2D
    model.add(Conv2D(filters=32, kernel_size=(3,3), strides=1, padding="same", activation='relu', input_shape=input_shape))
    # sử dụng max pooling
    model.add(MaxPool2D(pool_size=(2,2)))
    # thêm drop out để tránh overfitting (làm màu thôi, mạng này không overfitting được đâu :v)
    model.add(Dropout(0.5))
    # kéo phẳng các features từ 3D sang 2D
    model.add(Flatten())
    # tạo một mạng fully connected để phân loại ảnh
    model.add(Dense(units=128, activation='relu'))
    model.add(Dense(units=10, activation='softmax'))
    return model

my_model = create_model()
# biên dịch model
# bước biên dịch cho phép chúng ta lựa chọn loại hàm loss, thuật toán huấn luyện và thước đánh giá
# ở đây loss tôi chọn sparse_categorical_crossentropy vì chúng ta có nhãn là các số từ 0 đến 9
# và hàm activation của output là cross entropy
# tôi chọn optimizer là adam optimizer và metrics chắc chắn là accuracy.
my_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train/255
x_test = x_test/255
x_train = tf.reshape(x_train, (-1, 28, 28, 1))
x_test = tf.reshape(x_test, (-1, 28, 28, 1))
my_model.fit(x_train, y_train, batch_size=32, epochs=5, verbose=1, validation_data=(x_test, y_test))
score = my_model.evaluate(x_test, y_test, verbose=0)
print('Loss của tập test:', score[0])
print('Accuracy của tập test:', score[1])
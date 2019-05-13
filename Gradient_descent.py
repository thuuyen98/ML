from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

dataset= pd.read_csv("/Users/macos/Downloads/filted_train.csv")
dataset =dataset.fillna(dataset.mean())
dataset= dataset.replace('male', 0)
dataset= dataset.replace('female', 1)
features= dataset.iloc[:,1:].values
labels= dataset.iloc[:,:1].values
labels = np.squeeze(labels)

features_train, features_valid, labels_train, labels_valid = train_test_split(features, labels, test_size = 0.20)
print(features_train.shape)
print(features_valid.shape)
print(labels_train.shape)
print(labels_valid.shape)
lr = 0.0001
a = np.ones(shape=(6,1))
b = 0


def predict(x):
    return np.squeeze(np.matmul(x, a) + b)


# Tính đạo hàm loss theo a
def d_fa(X, Y, Y_pred):
    n = float(X.shape[0])
    return (-2 / n) * np.sum(np.matmul((Y - Y_pred), X), axis=0)


# Tính đạo hàm loss theo b
def d_fb(X, Y, Y_pred):
    n = float(X.shape[0])
    return (-2 / n) * np.sum(Y - Y_pred)


# Cập nhật giá trị mới cho a theo đạo hàm
def update_a(a, da):
    return a - lr * da


# Cập nhật giá trị mới cho b theo đạo hàm
def update_b(b, db):
    return b - lr * db


# Gradient Descent
def iris_gd(X_train, Y_train):
    global a, b
    iter_count = 0
    for iter_count in range(10000):
        train_pred = predict(X_train)
        da = d_fa(X_train, Y_train, train_pred)
        db = d_fb(X_train, Y_train, train_pred)
        a = update_a(a, da)
        b = update_b(b, db)


# Đánh giá mô hình
def eval(X_test, Y_test):
    predictions = predict(X_test)
    predictions = np.round(predictions)
    correct_pred = np.count_nonzero(predictions == Y_test)
    accuracy = correct_pred / predictions.shape[0]
    return accuracy


# Huấn luyện và đánh giá mô hình
#print(features_train.shape)
#print(labels_train.shape)
iris_gd(features_train, labels_train)
print("Ma trận trọng số a:\n", a)
print("Tham số b:", b)
acc = eval(features_valid, labels_valid)
print("Accuracy tập test:", acc)

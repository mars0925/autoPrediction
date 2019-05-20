# coding: utf-8
#使用自訂的模式

from LoadData import load_data
import numpy as np
import random
import pandas as pd

np.random.seed(10)
num_class = 2
RGB = 3  # 彩色
batchSize = 8
pixel = 224# 圖片的像素

# Step 1. 資料準備

(x_train, y_train), (x_test, y_test) = load_data()

# 打亂資料
index_1 = [i for i in range(len(x_train))]
random.shuffle(index_1)
x_train = x_train[index_1]
y_train = y_train[index_1]

index_2 = [i for i in range(len(x_test))]
random.shuffle(index_2)
x_test = x_test[index_2]
y_test = y_test[index_2]

print("train data:", 'images:', x_train.shape, " labels:", y_train.shape)
print("test data:", 'images:', x_test.shape, " labels:", y_test.shape)
 
# 正規化
x_train_normalize = x_train.astype('float32') / 255.0
x_test_normalize = x_test.astype('float32') / 255.0

from keras.utils import np_utils

y_train_OneHot = np_utils.to_categorical(y_train)
y_test_OneHot = np_utils.to_categorical(y_test)

print(y_train_OneHot.shape)
print(y_test_OneHot.shape)

# Step 2. 建立模型

from keras.models import Sequential # 初始化神經網路
from keras.layers import Dense  # 神經網路層 添加全連接層
from keras.layers import Dropout
from keras.layers import Flatten  # 扁平化
from keras.layers import Conv2D  # 卷積層
from keras.layers import MaxPooling2D  # Pooling layer 池化層

model = Sequential()  #初始化

# 卷積層1與池化層1

model.add(Conv2D(filters=32, kernel_size=(2, 2),
                 input_shape=(pixel, pixel, RGB),
                 activation='relu',
                 padding='same'))

model.add(Dropout(rate=0.5))

model.add(MaxPooling2D(pool_size=(2, 2)))

# 卷積層2與池化層2

model.add(Conv2D(filters=64, kernel_size=(2, 2),
                 activation='relu', padding='same'))

model.add(Dropout(0.5))

model.add(MaxPooling2D(pool_size=(2, 2)))


# Step 3. 建立神經網路(平坦層、隱藏層、輸出層)

model.add(Flatten())#扁平化 平坦層
model.add(Dropout(rate=0.5))

model.add(Dense(128, activation='relu'))#隱藏層
model.add(Dropout(rate=0.4))
model.add(Dense(256, activation='relu'))#隱藏層
model.add(Dropout(0.5))

model.add(Dense(num_class, activation='sigmoid'))  # 輸出層 有幾個類別 num_class

print(model.summary())

# 載入之前訓練的模型

try:
    model.load_weights("./bonscan.h5")
    print("載入模型成功!繼續訓練模型")
except:
    print("載入模型失敗!開始訓練一個新模型")

# Step 4. 訓練模型

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

train_history = model.fit(x_train_normalize, y_train_OneHot,
                          validation_split=0.2,
                          epochs=2, batch_size=batchSize, verbose=1,class_weight = 'auto')


def show_train_history(train_acc, test_acc):
    plt.plot(train_history.history[train_acc])
    plt.plot(train_history.history[test_acc])
    plt.title('Train History')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

import matplotlib.pyplot as plt
# 劃出訓練圖形
show_train_history('acc', 'val_acc')
show_train_history('loss', 'val_loss')

# =====#
# Step 6. 評估模型準確率
scores = model.evaluate(x_test_normalize, y_test_OneHot)
print("Loss:", scores[0], "accuracy", scores[1])


# 進行預測
# 利用訓練好的模型,用測試資料來預測他的類別
prediction = model.predict_classes(x_test_normalize)
prediction[:num_class]  # 類別數目

# 輸入標籤代表意義
label_dict = {0: "Normal ", 1: "Abnormal"}

Predicted_Probability = model.predict(x_test_normalize)#預測機率

print("＝＝＝＝＝＝＝列出測試集預測結果＝＝＝＝")

correct = 0

originalLabel = []# 原始標籤
predictlabel = []# 預測標籤
preProbability = []#預測機率

for i in range(y_test.shape[0]):
    originalLabel.append(label_dict[y_test[i]])
    predictlabel.append(label_dict[prediction[i]])
    preProbability.append(Predicted_Probability[i][np.argmax(Predicted_Probability[i], axis=0)])
    
    if prediction[i] == y_test[i]:
        correct += 1
        
print("Correct:", correct, " Total: ", len(y_test))


results = pd.DataFrame({"originalLabel": originalLabel,
                        "predictlabel": predictlabel,
                        "preProbability": preProbability})

results.to_csv("results23.csv", index=False)

# Step 8. Save Weight to h5

model.save_weights("./bonscan.h5")
print("Saved model to disk")

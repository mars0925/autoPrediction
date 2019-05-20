import os
import shutil

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import pandas as pd

predictPath = r"C:\Users\mars0925\Desktop\picture"  # 圖片資料夾的上一層
pixel = 224  # 預測圖片大小
num_class = 2  # 預測圖片類別
RGB = 3  # 彩色
predict_batchsize = 1  # 預測集一次學幾張圖片
label_dict = {0: 'abnormal', 1: 'normal'}  # 預測圖片種類的對應字典

# Initialising the CNN
model = Sequential()

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

model.add(Flatten())  # 扁平化 平坦層
model.add(Dropout(rate=0.5))

model.add(Dense(128, activation='relu'))  # 隱藏層
model.add(Dropout(rate=0.4))
model.add(Dense(256, activation='relu'))  # 隱藏層
model.add(Dropout(0.5))
model.add(Dense(num_class, activation='sigmoid'))  # 輸出層 有幾個類別 num_class

try:
    model.load_weights("./bonscan.h5")
    print("===Loading h5 success===")
except Exception as e:
    print(e)
    print("===Loading h5 error===")

predict_datagen = ImageDataGenerator(rescale=1. / 255)

predict_generator = predict_datagen.flow_from_directory(directory=predictPath,
                                                        target_size=(pixel, pixel),
                                                        color_mode="rgb",
                                                        batch_size=predict_batchsize,
                                                        class_mode=None,
                                                        shuffle=False)

prdictSize = predict_generator.n  # 預測集總張數
predict_step = int(prdictSize / predict_batchsize)

predict_generator.reset()
Predicted_Probability = model.predict_generator(predict_generator, steps=predict_step, verbose=1)
predicted_class_indices = np.argmax(Predicted_Probability, axis=1)

labels = {0: 'abnormal', 1: 'normal'}

probability = []  # 預測機率列表
for i in range(len(predicted_class_indices)):
    predict = Predicted_Probability[i][predicted_class_indices[i]]
    probability.append(predict)

predictions = [labels[k] for k in predicted_class_indices]

filenames = predict_generator.filenames
pcList = []

#取得檔案名稱
for item in filenames:
    pcList.append(item.split("\\")[-1])

results = pd.DataFrame({"檔案名稱": pcList,
                        "預測類別": predictions,
                        "預測機率": probability})

results.to_csv("results23.csv", index=False)

print("===分析完成===")

import time
time.sleep(0.25)

##把已經預測過的圖片移到其他資料夾
picPath = r"C:\Users\mars0925\Desktop\picture\predict" #圖片資料夾
backPath = r"C:\Users\mars0925\Desktop\backup" #備份資料夾

for index in range(len(pcList)):
    oldFilePath = os.path.join(picPath, pcList[index])  # 舊檔案絕對路徑
    newFilePath = os.path.join(backPath, pcList[index])  # 新檔案決斷路徑
    try:
        shutil.copy(oldFilePath, newFilePath)  # 複製過去
        os.remove(oldFilePath)  # 刪除
    except Exception as e:
        print(e)


print("===移檔完成===")
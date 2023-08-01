import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from read import rd, load, load_data
import h5py

# 图片文件夹路径
data_folder = './dataSet/photo_imbd_UTK/'
print("1")
images, labels = load(data_folder)
print("2")
# 划分数据集
X_train, X_temp, y_train, y_temp = train_test_split(images, labels, test_size=0.25, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.1, random_state=42)
print("3")
print(len(X_train), len(y_train))
print(len(X_val), len(y_val))
print(len(X_test), len(y_test))
print(X_train[0])
print(type(X_train[0]))

#vg16

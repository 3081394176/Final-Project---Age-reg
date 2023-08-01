import os
import cv2
import numpy as np
from tqdm import tqdm
from save import save_data, load_data
import h5py
import pickle


def rd(data_folder):
    # 图片和标签列表
    images = []
    labels = []

    # 遍历文件夹
    for file_name in tqdm(os.listdir(data_folder)):
        # 分割文件名以获得年龄标签
        label = file_name.split('_')[0]

        # 读取图像
        img = cv2.imread(os.path.join(data_folder + file_name), cv2.IMREAD_COLOR)
        target_size = (256, 256)
        if img is None:
            print(f"Unable to load image: {file_name}")
            continue
        img = cv2.resize(img, target_size)

        # 检查图片形状
        if img.shape != (256, 256, 3):
            print(f"Unexpected image shape: {img.shape} for image: {file_name}")

        # 将图像和标签添加到列表
        images.append(img)
        labels.append(float(label))  # 确保标签为浮点数

    # 将列表转化为numpy数组
    #images = np.array(images, dtype=np.uint8)  # 使用dtype=object来处理形状不一样的数据

    return np.stack(images), np.array(labels, dtype=float)

def load_data_from_pickle(file_name):
    """
    从.pkl文件中加载数据，并返回加载的数据。

    Parameters:
        file_name (str): .pkl文件的路径和文件名。

    Returns:
        data: 加载的数据。
    """
    with open(file_name, 'rb') as f:
        data = pickle.load(f)
    return data

def load(data_folder):
    if(load_data("images.pkl") is None or load_data("labels.pkl") is None):
        images, labels = rd(data_folder)
        save_data(images, "images.pkl")
        save_data(labels, "labels.pkl")

    return load_data_from_pickle("images.pkl"), load_data_from_pickle("labels.pkl")


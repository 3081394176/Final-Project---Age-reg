import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from read import rd, load, load_data

# 图片文件夹路径
data_folder = './dataSet/photo_imbd_UTK/'

images, labels = load(data_folder)
print("255")
images = images / 255.0
print("255done")
print("set")
# 划分数据集
X_train, X_temp, y_train, y_temp = train_test_split(images, labels, test_size=0.25, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.1, random_state=42)
print("set done")
import os
from keras.applications import ResNet50
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model, load_model
import numpy as np
import matplotlib.pyplot as plt

# 检查模型文件是否存在
model_path = './model/cnn_model.h5'
if os.path.exists(model_path):
    # 加载模型
    model = load_model(model_path)
else:
    # 创建模型
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(256, 256, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(1, activation='linear')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer='adam', loss='mean_squared_error')

    # 训练模型
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)

    # 保存模型
    if not os.path.exists('./model'):
        os.makedirs('./model')
    model.save(model_path)

# 预测并评估
y_pred = model.predict(X_test)
mse = np.mean((y_test - y_pred) ** 2)
print(f"Mean Squared Error: {mse}")

# 可视化预测与真实值
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, c='blue')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', c='red')
plt.xlabel('True Age')
plt.ylabel('Predicted Age')
plt.title('True Age vs Predicted Age')
plt.show()


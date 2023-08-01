import os
import cv2
import dlib
from tqdm import tqdm
from datetime import datetime

# 设置dlib人脸检测器
detector = dlib.get_frontal_face_detector()

# 设置源目录和目标目录
source_dir = './dataSet/imdb_crop'
dest_dir = './dataSet/filted_photos'
os.makedirs(dest_dir, exist_ok=True)

# 获取源目录及其子目录中所有文件
all_files = []
for root, dirs, files in os.walk(source_dir):
    all_files.extend([os.path.join(root, f) for f in files if f.endswith('.jpg')])

# 创建进度条
progress_bar = tqdm(total=len(all_files), desc='Processing images')

# 遍历源目录及其子目录中的所有文件
for file_path in all_files:
    # 更新进度条
    progress_bar.update(1)

    # 读取图片
    img = cv2.imread(file_path)

    # 人脸检测
    dets = detector(img, 1)

    # 如果检测到的人脸只有一张，并且图像尺寸足够大，将其保存到目标目录
    if len(dets) == 1 and img.shape[0] > 200 and img.shape[1] > 200:
        # 提取出生日期和照片拍摄日期
        filename = os.path.basename(file_path)
        birth_date_str = filename.split('_')[2]
        photo_date_str = filename.split('_')[3].split('.')[0]

        # 检查日期格式，如果格式错误则跳过该图片
        try:
            # 转换为日期对象
            birth_date = datetime.strptime(birth_date_str, '%Y-%m-%d')
            photo_date = datetime.strptime(photo_date_str, '%Y')

            # 计算精确年龄，结果是一个小数
            age = (photo_date - birth_date).days / 365.25

            # 设置新的文件名，保留年龄信息
            new_filename = f"{filename.split('_')[0]}_{age:.2f}.jpg"

            cv2.imwrite(os.path.join(dest_dir, new_filename), img)
        except ValueError:
            print(f"Skipping file {filename} due to invalid date format.")
            continue

# 关闭进度条
progress_bar.close()

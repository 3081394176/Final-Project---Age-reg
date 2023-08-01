import os
from PIL import Image
from tqdm import tqdm

def process_images(input_folder, output_folder):
    age_count_dict = {}  # 新建一个字典来跟踪每个年龄的图片数量

    filenames = os.listdir(input_folder)
    for filename in tqdm(filenames, desc=f"Processing images from {input_folder}"):
        # 只处理图片文件，排除其他类型的文件
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        img_path = os.path.join(input_folder, filename)
        try:
            img = Image.open(img_path)
        except IOError:
            continue

        # 获取图片的宽度和高度
        width, height = img.size

        # 如果不是等宽，则裁剪成等宽
        if width != height:
            min_dim = min(width, height)
            left = (width - min_dim) / 2
            top = (height - min_dim) / 2
            right = (width + min_dim) / 2
            bottom = (height + min_dim) / 2

            img = img.crop((left, top, right, bottom))

        # 如果像素大于256x256，缩小图片
        if img.width > 256 and img.height > 256:
            img = img.resize((256, 256), Image.LANCZOS)

        # 如果像素小于256x256，放大图片
        if img.width < 256 and img.height < 256:
            img = img.resize((256, 256), Image.BICUBIC)

        # 获取年龄信息
        if 'UTKFace' in input_folder:
            age = filename.split('_')[0]
        elif 'filted_photos' in input_folder:
            age = filename.rsplit('_', 1)[1].replace('.jpg', '')
        else:
            print("Unknown folder!")
            continue

        # 从字典中获取当前年龄的图片数量
        count = age_count_dict.get(age, 0) + 1

        # 更新字典
        age_count_dict[age] = count

        # 保存处理过的图片
        save_name = f"{output_folder}/{age}_{count}.jpg"
        img.save(save_name)

def main():
    input_folders = ['./dataSet/UTKFace', './dataSet/filted_photos']
    output_folder = './dataSet/photo_imbd_UTK'

    # 创建输出文件夹，如果不存在
    os.makedirs(output_folder, exist_ok=True)

    for input_folder in input_folders:
        process_images(input_folder, output_folder)

if __name__ == "__main__":
    main()

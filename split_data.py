import os
import shutil
import random

# 设置随机种子以保证复现性
random.seed(42)

# 原始数据集路径
source_dir = 'F:/Knee/ultralytics-main/MedicalExpert-I'  # 改成你五个分类文件夹所在的路径
# 目标路径
train_dir = os.path.join(source_dir, 'train')
val_dir = os.path.join(source_dir, 'val')

# 创建 train 和 val 文件夹及其子类别文件夹
for split_dir in [train_dir, val_dir]:
    os.makedirs(split_dir, exist_ok=True)

# 遍历每一个类别
for category in os.listdir(source_dir):
    category_path = os.path.join(source_dir, category)
    if not os.path.isdir(category_path) or category in ['train', 'val']:
        continue

    # 获取该类别所有图像文件
    images = os.listdir(category_path)
    random.shuffle(images)

    # 划分索引
    split_idx = int(0.8 * len(images))
    train_images = images[:split_idx]
    val_images = images[split_idx:]

    # 创建子目录
    os.makedirs(os.path.join(train_dir, category), exist_ok=True)
    os.makedirs(os.path.join(val_dir, category), exist_ok=True)

    # 移动或复制图像
    for img in train_images:
        shutil.copy(os.path.join(category_path, img),
                    os.path.join(train_dir, category, img))
    for img in val_images:
        shutil.copy(os.path.join(category_path, img),
                    os.path.join(val_dir, category, img))

print("数据集划分完成！")
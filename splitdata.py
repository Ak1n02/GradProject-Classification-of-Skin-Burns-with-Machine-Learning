import os
import shutil
import random

# Paths
data_dir = "raw"
output_dir = "raw_splitted"

# Classes
categories = ["Ak1n02 bg-rem test_eren First_Degree_removed", "Ak1n02 bg-rem test_eren Second_Degree_removed", "Ak1n02 bg-rem test_eren Third_Degree_removed"]

# Split ratios
train_ratio, val_ratio, test_ratio = 0.7, 0.15, 0.15

# Create train, val, test directories
for split in ["train", "val", "test"]:
    for category in categories:
        os.makedirs(os.path.join(output_dir, split, category), exist_ok=True)

# Split dataset
for category in categories:
    img_list = os.listdir(os.path.join(data_dir, category))
    random.shuffle(img_list)

    train_split = int(len(img_list) * train_ratio)
    val_split = int(len(img_list) * (train_ratio + val_ratio))

    train_imgs = img_list[:train_split]
    val_imgs = img_list[train_split:val_split]
    test_imgs = img_list[val_split:]

    # Copy images
    for img in train_imgs:
        shutil.copy(os.path.join(data_dir, category, img), os.path.join(output_dir, "train", category, img))

    for img in val_imgs:
        shutil.copy(os.path.join(data_dir, category, img), os.path.join(output_dir, "val", category, img))

    for img in test_imgs:
        shutil.copy(os.path.join(data_dir, category, img), os.path.join(output_dir, "test", category, img))

print("Dataset successfully split into train, val, and test!")

# Python Standard Library Imports
import os
import time
from pathlib import Path as PathlibPath
from os.path import isfile, join

# Third-Party Imports
import numpy as np
from PIL import Image
import imageio

# Constants
IMAGE_DIR_NAME = "image_jiangtong_intersection_2_8_4"
# IMAGE_DIR_NAME = "image_jiangtong_intersection_2_8_4"
# 自动计算路径
BASE_DIR = PathlibPath(__file__).parent
PNGS_DIR = BASE_DIR  /IMAGE_DIR_NAME  / "images"
GIF_FILE_PATH = BASE_DIR  /IMAGE_DIR_NAME / f"{IMAGE_DIR_NAME}.gif"
IMAGE_TEMPLATE = f'{IMAGE_DIR_NAME}_{{}}.png'

def generate_gif_from_images():
    # 记录gif生成时间,用于评估效率,没有特殊用途
    tic = time.time()
    
    # 创建一个图像列表,用于存储所有图像
    image_list = []
    
    # 获取目录下PNG图像文件的列表
    if not os.path.exists(PNGS_DIR):
        print(f"路径 {PNGS_DIR} 不存在!")
    
    files_in_directory = [file for file in PNGS_DIR.iterdir() if file.is_file() and file.name.endswith(".png")]
    
    # 找到文件夹中存在的最大编号
    max_image_index = -1
    for file in files_in_directory:
        try:
            index = int(file.stem.split('_')[-1])
            max_image_index = max(max_image_index,index)
        except ValueError:
            pass
    
    # 循环读取图片并添加到列表中
    for i in range(max_image_index + 1):
        image_path = PNGS_DIR / IMAGE_TEMPLATE.format(i)
        if image_path.exists():
            img = Image.open(image_path)
            image_list.append(img)
    
    # 保存GIF动画
    if image_list:
        image_list[0].save(GIF_FILE_PATH,save_all=True,append_images=image_list[1:],duration=100,loop=0)  # duration是指每一帧在gif中的显示时间（停留时间）
    
    print(f"###log### GIF文件保存在 {GIF_FILE_PATH}")
    toc = time.time()
    print(f"###log### GIF generated in {toc - tic:.2f} seconds.")



if __name__ == "__main__":
    generate_gif_from_images()

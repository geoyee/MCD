import os
import os.path as osp
import random
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
from utils import cut_image, pre_togray


def save_palette(label, save_path):
    bin_colormap = np.ones((256, 3)) * 255
    bin_colormap[0, :] = [0, 0, 0]
    bin_colormap = bin_colormap.astype(np.uint8)
    visualimg  = Image.fromarray(label, "P")
    palette = bin_colormap
    visualimg.putpalette(palette) 
    visualimg.save(save_path, format="PNG")


if __name__ == "__main__":
    # 1.设置
    re_name = True  # 是否重命名
    idx_offset = 0  # 重命名偏移
    is_cut = True  # 是否裁剪
    is_rand = True  # 是否乱序
    data_dir = r"E:\MCD\dataset"
    image_mid_path = "JPEGImages"
    label_mid_path = "Annotations"
    # 2.处理标签及重用名
    names = os.listdir(osp.join(data_dir, image_mid_path))
    if is_rand:
        random.shuffle(names)
    for idx, name in tqdm(enumerate(names, start=1)):
        image_path = osp.join(data_dir, image_mid_path, name)
        label_path = osp.join(data_dir, label_mid_path, name.replace(".jpg", ".png"))
        if osp.exists(label_path):
            name_without_ext = name.split(".")[0]
            if re_name:
                new_name_without_ext = "pre" + str(idx_offset + idx)
                image_save_path = image_path.replace(name_without_ext, new_name_without_ext)
                label_save_path = label_path.replace(name_without_ext, new_name_without_ext)
            else:
                image_save_path = image_path
                label_save_path = label_path
            # 3.保存图像
            image = cv2.imread(image_path)
            image = cut_image(image)
            os.remove(image_path)
            cv2.imwrite(image_save_path, image)
            # 4.保存标签
            label = cv2.imread(label_path)
            label = cut_image(label)
            label = pre_togray(label)
            os.remove(label_path)
            save_palette(label, label_save_path)
        else:
            os.remove(image_path)

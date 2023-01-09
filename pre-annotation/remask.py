import os
import os.path as osp
import random
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


def label_deal(lab):
    num_shp = len(lab.shape)
    if num_shp == 2:
        pass
    elif num_shp == 3:
        lab = cv2.cvtColor(lab, cv2.COLOR_BGR2GRAY)
    elif num_shp == 4:
        lab = cv2.cvtColor(lab, cv2.COLOR_BGRA2GRAY)
    else:
        raise ValueError("ERROR")
    result = np.zeros_like(lab)
    result[lab != 0] = 1
    return result.astype("uint8")


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
    data_dir = r"E:\MyData\graduate\BS\Crack\.crack\infs\pretrain"
    image_mid_path = "JPEGImages"
    label_mid_path = "Annotations"
    # 2.处理标签及重用名
    names = os.listdir(osp.join(data_dir, image_mid_path))
    random.shuffle(names)
    for idx, name in tqdm(enumerate(names, start=1)):
        image_path = osp.join(data_dir, image_mid_path, name)
        label_path = osp.join(data_dir, label_mid_path, name.replace(".jpg", ".png"))
        if osp.exists(label_path):
            name_without_ext = name.split(".")[0]
            new_name_without_ext = "c" + str(idx)
            image_save_path = image_path.replace(name_without_ext, new_name_without_ext)
            label_save_path = label_path.replace(name_without_ext, new_name_without_ext)
            label = cv2.imread(label_path)
            label = label_deal(label)
            os.rename(image_path, image_save_path)
            os.remove(label_path)
            save_palette(label, label_save_path)
        else:
            os.remove(image_path)

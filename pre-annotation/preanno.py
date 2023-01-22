import os
import os.path as osp
import cv2
import numpy as np
import paddle
from PIL import Image
from tqdm import tqdm
from utils import (
    cut_image, init_model, val_transforms, 
    pre_togray, erode, deal_connection
)


if __name__ == "__main__":
    # 1.设置
    image_dir = r"E:\MyData\graduate\BS\Crack\.crack\infs\DIY\New\JPEGImages"
    output_dir = r"E:\MyData\graduate\BS\Crack\.crack\infs\DIY\New\output"
    # 2.创建文件夹
    img_save_dir = osp.join(output_dir, "JPEGImages")
    os.makedirs(img_save_dir, exist_ok=True)
    lab_save_dir = osp.join(output_dir, "Annotations")
    os.makedirs(lab_save_dir, exist_ok=True)
    model = init_model()
    # 3.预处理及保存数据
    names = os.listdir(image_dir)
    for name in tqdm(names):
        image_path = osp.join(image_dir, name)
        name_without_ext = name.split(".")[0]
        img_save_path = osp.join(img_save_dir, (name_without_ext + ".jpg"))
        lab_save_path = osp.join(lab_save_dir, (name_without_ext + ".png"))
        # 1.预处理图像
        image = cv2.imread(image_path)
        image = cut_image(image)
        cv2.imwrite(img_save_path, image)
        # 2.预标注
        img, _ = val_transforms(img_save_path)
        img = paddle.to_tensor(img[np.newaxis, :])
        pred = paddle.argmax(model(img)[0], axis=1).squeeze().numpy().astype("uint8")
        # 3.后处理
        pred = pre_togray(pred)
        # pred = erode(pred)
        pred = deal_connection(pred)
        # 4.保存标签
        Image.fromarray(pred).save(lab_save_path, "PNG")

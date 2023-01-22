import os
import os.path as osp
import cv2
from tqdm import tqdm


def cut_image(img, n_size=(480, 320)):
    n_w, n_h = n_size
    # 1.图像旋转
    h, w = img.shape[:2]
    if h > w:
        img = cv2.transpose(img)
    h, w = img.shape[:2]
    # 2.图像缩放
    h_rate = n_h / h
    w_rate = n_w / w
    rate = w_rate if w_rate > h_rate else h_rate
    img = cv2.resize(img, (int(w * rate), int(h * rate)), interpolation=cv2.INTER_CUBIC)
    # 3.中心裁剪
    c_h, c_w = img.shape[:2]
    c_h /= 2
    c_w /= 2
    x = int(c_w - n_w / 2)
    y = int(c_h - n_h / 2)
    crop_img = img[y: y + n_h, x: x + n_w, :]
    return crop_img


if __name__ == "__main__":
    # 1.设置
    image_dir = r"E:\MyData\graduate\BS\Crack\.crack\infs\DIY\New\JPEGImages"
    output_dir = r"E:\MyData\graduate\BS\Crack\.crack\infs\DIY\New\JPEGImages2"
    # 2.创建文件夹
    os.makedirs(output_dir, exist_ok=True)
    # 3.预处理及保存数据
    names = os.listdir(image_dir)
    for name in tqdm(names):
        image_path = osp.join(image_dir, name)
        save_path = osp.join(output_dir, name)
        image = cv2.imread(image_path)
        image = cut_image(image)
        cv2.imwrite(save_path, image)

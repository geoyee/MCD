import os
import os.path as osp
import cv2
import numpy as np
from tqdm import tqdm


def pre_togray(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    result = np.zeros_like(img)
    result[img != 0] = 1
    return result.astype("uint8")


def deal_connection(pred, threshold=8):
    result = np.zeros_like(pred)
    contours, reals = cv2.findContours(pred, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if len(contours) == 0:
        return pred
    for contour, real in zip(contours, reals[0]):
        if real[-1] == -1:  # 没有父轮廓
            if cv2.contourArea(contour) > threshold:
                cv2.fillPoly(result, [contour], (255))
        else:
            cv2.fillPoly(result, [contour], (0))
    return result.astype("uint8")


if __name__ == "__main__":
    # 1.设置
    image_dir = r"E:\MyData\graduate\BS\Crack\.crack\infs\DIY\Annotations2"
    # 2.处理小联通区
    names = os.listdir(image_dir)
    for idx, name in tqdm(enumerate(names, start=1)):
        image_path = osp.join(image_dir, name)
        image = cv2.imread(image_path)
        image = pre_togray(image)
        image = deal_connection(image)
        os.remove(image_path)
        cv2.imwrite(image_path, image)

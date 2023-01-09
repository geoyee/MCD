import os
import os.path as osp
import cv2
import numpy as np
from tqdm import tqdm


def pre_togray(lab):
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


def erode(img):
    kernel = np.ones((2, 2), np.uint8)
    erosion = cv2.erode(img, kernel, iterations=1)
    return erosion


def deal_connection(pred, threshold=8):
    result = np.zeros_like(pred)
    contours, reals = cv2.findContours(pred, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if len(contours) == 0:
        return pred
    for contour, real in zip(contours, reals[0]):
        if real[-1] == -1:
            if cv2.contourArea(contour) > threshold:
                cv2.fillPoly(result, [contour], (255))
        else:
            cv2.fillPoly(result, [contour], (0))
    return result.astype("uint8")


if __name__ == "__main__":
    # 1.设置
    image_dir = r"E:\MyData\graduate\BS\Crack\.crack\infs\DIY\Annotations"
    # 2.处理小联通区
    names = os.listdir(image_dir)
    for idx, name in tqdm(enumerate(names, start=1)):
        image_path = osp.join(image_dir, name)
        image = cv2.imread(image_path)
        image = pre_togray(image)  # 1.预处理
        image = erode(image)  # 2.腐蚀
        image = deal_connection(image)  # 3.删除小的联通区
        os.remove(image_path)
        cv2.imwrite(image_path, image)

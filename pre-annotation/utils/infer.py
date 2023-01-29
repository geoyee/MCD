import os
import os.path as osp
import numpy as np
import paddle
import paddleseg.transforms as T
from PIL import Image
from paddleseg.models import OCRNet, HRNet_W48


def init_model(url="pre-annotation/utils/model.pdparams"):
    model = OCRNet(
        num_classes=2,
        backbone=HRNet_W48(),
        backbone_indices=[0]
    )
    model.set_state_dict(paddle.load(url))
    model.eval()
    return model


val_transforms = T.Compose([T.Normalize()])


if __name__ == "__main__":
    imgs_dir = r"E:\MCD\dataset\JPEGImages"
    save_dir = r"E:\MCD\dataset\JPEGImages2"
    model = init_model()
    if not osp.exists(save_dir):
        os.makedirs(save_dir)
    img_names = os.listdir(imgs_dir)
    for name in img_names:
        img_path = osp.join(imgs_dir, name)
        img, _ = val_transforms(img_path)
        img = paddle.to_tensor(img[np.newaxis, :])
        pred = paddle.argmax(model(img)[0], axis=1).squeeze().numpy().astype("uint8")
        Image.fromarray(pred * 255).save(osp.join(save_dir, name.replace(".jpg", ".png")), "PNG")

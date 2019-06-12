import os
import cv2
import numpy as np
import torch
from skimage import transform as sktsf
from torchvision import transforms as tvtsf
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset


def normalize(img):
    """
    https://github.com/pytorch/vision/issues/223
    return appr -1~1 RGB
    """
    normalizing = tvtsf.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
    img = normalizing(torch.from_numpy(img).float())
    return img.numpy()


def resize_bbox(bbox, in_size, out_size):
    """Resize bounding boxes according to image resize.
    The bounding boxes are expected to be packed into a two dimensional
    tensor of shape :math:`(R, 4)`, where :math:`R` is the number of
    bounding boxes in the image. The second axis represents attributes of
    the bounding box. They are :math:`(y_{min}, x_{min}, y_{max}, x_{max})`,
    where the four attributes are coordinates of the top left and the
    bottom right vertices.
    Args:
        bbox (~numpy.ndarray): An array whose shape is :math:`(R, 4)`.
            :math:`R` is the number of bounding boxes.
        in_size (tuple): A tuple of length 2. The height and the width
            of the image before resized.
        out_size (tuple): A tuple of length 2. The height and the width
            of the image after resized.
    Returns:
        ~numpy.ndarray:
        Bounding boxes rescaled according to the given image shapes.
    """
    bbox = bbox.copy()
    y_scale = float(out_size[0]) / in_size[0]
    x_scale = float(out_size[1]) / in_size[1]
    bbox[:, 0] = y_scale * bbox[:, 0]
    bbox[:, 2] = y_scale * bbox[:, 2]
    bbox[:, 1] = x_scale * bbox[:, 1]
    bbox[:, 3] = x_scale * bbox[:, 3]
    return bbox.astype(int)


# https://github.com/chenyuntc/simple-faster-rcnn-pytorch/blob/master/data/dataset.py#L42
def preprocess(img, min_size=600, max_size=1000):
    """Preprocess an image for feature extraction.
    The length of the shorter edge is scaled to :obj:`self.min_size`.
    After the scaling, if the length of the longer edge is longer than
    :param min_size:
    :obj:`self.max_size`, the image is scaled to fit the longer edge
    to :obj:`self.max_size`.
    After resizing the image, the image is subtracted by a mean image value
    :obj:`self.mean`.
    Args:
        img (~numpy.ndarray): An image. This is in CHW and RGB format.
            The range of its value is :math:`[0, 255]`.
    Returns:
        ~numpy.ndarray: A preprocessed image.
    """
    C, H, W = img.shape
    scale1 = min_size / min(H, W)
    scale2 = max_size / max(H, W)
    scale = min(scale1, scale2)
    img = img / 255.
    img = sktsf.resize(img, (C, H * scale, W * scale), mode='reflect', anti_aliasing=False)
    # both the longer and shorter should be less than
    # max_size and min_size

    return normalize(img)


class Transform:

    def __init__(self, min_size=600, max_size=1000):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, in_data):
        img, bbox, label = in_data
        _, H, W = img.shape
        img = preprocess(img, self.min_size, self.max_size)
        _, o_H, o_W = img.shape
        scale = o_H / H
        bbox = resize_bbox(bbox, (H, W), (o_H, o_W))

        return img, bbox, label, scale


# https://github.com/chenyuntc/simple-faster-rcnn-pytorch/blob/0.4/data/dataset.py
# https://github.com/chainer/chainercv/blob/v0.12.0/chainercv/datasets/voc/voc_bbox_dataset.py#L11
class LoadDataset(Dataset):
    def __init__(self, data_dir, split='trainval', use_difficult=False, return_difficult=False, train=True,
                 transform=Transform()):
        # data_pth = /VOCdevkit/VOC2007
        id_list_file = os.path.join(data_dir, 'ImageSets/Main/{}.txt'.format(split))
        assert os.path.exists(id_list_file) == True, "file:{} not found.".format(id_list_file)
        with open(id_list_file, 'r') as f:
            id_lines = f.read().splitlines()
        self.ids = id_lines
        self.data_dir = data_dir
        self.use_difficult = use_difficult
        self.return_difficult = return_difficult
        self.train = train
        self.transform = transform
        VOC_BBOX_LABEL_NAMES = (
            'aeroplane',
            'bicycle',
            'bird',
            'boat',
            'bottle',
            'bus',
            'car',
            'cat',
            'chair',
            'cow',
            'diningtable',
            'dog',
            'horse',
            'motorbike',
            'person',
            'pottedplant',
            'sheep',
            'sofa',
            'train',
            'tvmonitor')
        self.label_names = VOC_BBOX_LABEL_NAMES

    def __len__(self):
        return len(self.ids)

    def _get_image(self, i):
        img_path = os.path.join(self.data_dir, 'JPEGImages', i + '.jpg')
        img = cv2.imread(img_path)  # BGR, (333, 500, 3)
        img = np.transpose(img, (2, 0, 1))  # # RBG, (3, 333, 500)
        assert img is not None, 'File Not Found: {}'.format(img_path)
        return img

    def _get_annotations(self, i):
        anno = ET.parse(os.path.join(self.data_dir, 'Annotations', i + '.xml'))
        bbox, label, difficult = [], [], []

        for obj in anno.findall('object'):
            # when in not using difficult split, and the object is
            # difficult, skipt it.
            if not self.use_difficult and int(obj.find('difficult').text) == 1:
                continue

            difficult.append(int(obj.find('difficult').text))
            bndbox_anno = obj.find('bndbox')
            bbox.append([int(bndbox_anno.find(tag).text) - 1 for tag in ('xmin', 'ymin', 'xmax', 'ymax')])
            name = obj.find('name').text.lower().strip()
            label.append(self.label_names.index(name))

        bbox = np.stack(bbox).astype(np.float32)
        label = np.stack(label).astype(np.int32)
        difficult = np.array(difficult, dtype=np.bool).astype(np.uint8)

        return bbox, label, difficult

    def __getitem__(self, idx):
        indx = self.ids[idx]
        img = self._get_image(indx)
        bbox, label, difficult = self._get_annotations(indx)
        if self.train:
            img, bbox, label, scale = self.transform((img, bbox, label))

        return img, bbox, label, difficult



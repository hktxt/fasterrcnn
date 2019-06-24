from __future__ import absolute_import
from __future__ import division
import torch as t
import cv2
import os
from imageio import imread
from utils.util import im_list_to_blob
from torch.utils.data import Dataset
from utils.voc_dataset import VOCBboxDataset
from skimage import transform as sktsf
from torchvision import transforms as tvtsf
from utils import util
import numpy as np
from utils.config import opt


def inverse_normalize(img):
    if opt.caffe_pretrain:
        img = img + (np.array([122.7717, 115.9465, 102.9801]).reshape(3, 1, 1))
        return img[::-1, :, :]
    # approximate un-normalize for visualize
    return (img * 0.225 + 0.45).clip(min=0, max=1) * 255


def pytorch_normalze(img):
    """
    https://github.com/pytorch/vision/issues/223
    return appr -1~1 RGB
    """
    normalize = tvtsf.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
    img = normalize(t.from_numpy(img))
    return img.numpy()


def caffe_normalize(img):
    """
    return appr -125-125 BGR
    """
    img = img[[2, 1, 0], :, :]  # RGB-BGR
    img = img * 255
    mean = np.array([122.7717, 115.9465, 102.9801]).reshape(3, 1, 1)
    img = (img - mean).astype(np.float32, copy=True)
    return img


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
    if opt.caffe_pretrain:
        normalize = caffe_normalize
    else:
        normalize = pytorch_normalze
    return normalize(img)


class Transform(object):

    def __init__(self, min_size=600, max_size=1000, filp=True):
        self.min_size = min_size
        self.max_size = max_size
        self.filp = filp

    def __call__(self, in_data):
        img, bbox, label = in_data
        _, H, W = img.shape
        img = preprocess(img, self.min_size, self.max_size)
        _, o_H, o_W = img.shape
        scale = o_H / H
        bbox = util.resize_bbox(bbox, (H, W), (o_H, o_W))

        # horizontally flip
        if self.filp:
            img, params = util.random_flip(
                img, x_random=True, return_param=True)
            bbox = util.flip_bbox(
                bbox, (o_H, o_W), x_flip=params['x_flip'])

        return img, bbox, label, scale


class Dataset:
    def __init__(self, opt, split, filp=True):
        self.opt = opt
        self.filp = filp
        self.db = VOCBboxDataset(opt.voc_data_dir, split=split)
        self.tsf = Transform(opt.min_size, opt.max_size, filp=self.filp)

    def __getitem__(self, idx):
        ori_img, bbox, label, difficult = self.db.get_example(idx)
        # bbox: (ymin, xmin, ymax, xmax) - 1

        img, bbox, label, scale = self.tsf((ori_img, bbox, label))

        # suit for Net input
        im_info = np.array((img.shape[1], img.shape[2], scale), dtype=np.float32)
        gt_boxes = np.append(bbox, label[:, np.newaxis], axis=1).astype(np.float32)
        # ymin, xmin, ymax, xmax -> xmin ymin, xmax, ymax
        gt_boxes[:, [0, 1, 2, 3]] = gt_boxes[:, [1, 0, 3, 2]]
        num_boxes = gt_boxes.shape[0]
        # fix some of the strides of a given numpy array are negative.
        # https://discuss.pytorch.org/t/torch-from-numpy-not-support-negative-strides/3663
        return img.copy(), im_info.copy(), gt_boxes.copy(), num_boxes

        # return img.copy(), bbox.copy(), label.copy(), scale

    def __len__(self):
        return len(self.db)


class TestDataset:
    def __init__(self, opt, split='test', use_difficult=True):
        self.opt = opt
        self.db = VOCBboxDataset(opt.voc_data_dir, split=split, use_difficult=use_difficult)

    def __getitem__(self, idx):
        ori_img, bbox, label, difficult = self.db.get_example(idx)
        _, H, W = ori_img.shape
        img = preprocess(ori_img)
        _, o_H, o_W = img.shape
        scale = o_H / H

        # suit for Net input
        im_info = np.array((img.shape[1], img.shape[2], scale), dtype=np.float32)
        gt_boxes = np.append(bbox, label[:, np.newaxis], axis=1).astype(np.float32)
        num_boxes = gt_boxes.shape[0]
        # fix some of the strides of a given numpy array are negative.
        # https://discuss.pytorch.org/t/torch-from-numpy-not-support-negative-strides/3663
        return img.copy(), im_info.copy(), gt_boxes.copy(), num_boxes
        #return img, ori_img.shape[1:], bbox, label, difficult

    def __len__(self):
        return len(self.db)


class CusDataset(Dataset):
    def __init__(self, path='E:/condaDev/fasterrcnn/myimplemention/samples/'):
        self.path = path
        self.images = os.listdir(self.path)
        assert len(self.images) > 0, 'No images found in {}'.format(self.path )

    def __len__(self):
        return len(self.images)

    def _get_image_blob(self, im):
        """Converts an image into a network input.
        Arguments:
        im (ndarray): a color image in BGR order
        Returns:
        blob (ndarray): a data blob holding an image pyramid
        im_scale_factors (list): list of image scales (relative to im) used
          in the image pyramid
        """
        im_orig = im.astype(np.float32, copy=True)
        im_orig -= opt.PIXEL_MEANS

        im_shape = im_orig.shape
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])

        processed_ims = []
        im_scale_factors = []

        for target_size in opt.TEST_SCALES:
            im_scale = float(target_size) / float(im_size_min)
            # Prevent the biggest axis from being more than MAX_SIZE
            if np.round(im_scale * im_size_max) > opt.TEST_MAX_SIZE:
                im_scale = float(opt.TEST_MAX_SIZE) / float(im_size_max)
            im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
            im_scale_factors.append(im_scale)
            processed_ims.append(im)

        # Create a blob to hold the input images
        blob = im_list_to_blob(processed_ims)[0]

        return blob, np.array(im_scale_factors, dtype=np.float32)

    def __getitem__(self, idx):
        img_pth = self.path + self.images[idx]
        im_in = np.array(imread(img_pth))
        if len(im_in.shape) == 2:
            im_in = im_in[:, :, np.newaxis]
            im_in = np.concatenate((im_in, im_in, im_in), axis=2)
        # rgb -> bgr
        im = im_in[:, :, ::-1]
        blobs, im_scales = self._get_image_blob(im)
        im_blob = blobs
        assert len(im_scales) == 1, "Only single-image batch implemented"
        im_info_np = np.array([im_blob.shape[1], im_blob.shape[2], im_scales[0]], dtype=np.float32)
        im_data = np.transpose(im_blob, (2, 1, 0))
        gt_boxes = np.ones([1,5], dtype=np.float32)
        num_boxes = np.array(1)
        return im.copy(), im_data.copy(), im_info_np.copy(), gt_boxes.copy(), num_boxes, im_scales, self.images[idx]



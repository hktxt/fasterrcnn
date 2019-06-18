# https://github.com/ruotianluo/pytorch-faster-rcnn/blob/master/lib/nets/network.py#L97
from torchvision.ops import RoIAlign, RoIPool
from utils.config import opt


def roi_pool_layer(bottom, rois):
    return RoIPool((opt.pooling_size, opt.pooling_size),
                   1.0 / opt.feat_stride)(bottom, rois)


def roi_align_layer(bottom, rois):
    return RoIAlign((opt.pooling_size, opt.pooling_size), 1.0 / opt.feat_stride,
                    0)(bottom, rois)

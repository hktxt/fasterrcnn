# from https://github.com/chenyuntc/simple-faster-rcnn-pytorch/blob/master/utils/config.py
from pprint import pprint


# Default Configs for training
# NOTE that, config items could be overwriten by passing argument through command line.
# e.g. --voc-data-dir='./data/'

class Config:
    # data
    voc_data_dir = 'E:/data/voc07/VOCdevkit/VOC2007'
    min_size = 600   # image resize
    max_size = 1000  # image resize
    disp_interval = 100
    test_num_workers = 0

    # sigma for l1_smooth_loss
    rpn_sigma = 3.
    roi_sigma = 1.

    # resume
    resume = False

    logs = 'logs'
    save_dir = 'output'

    # param for optimizer
    # 0.0005 in origin paper but 0.0001 in tf-faster-rcnn
    weight_decay = 0.0005
    lr_decay_gamma = 0.1  # 1e-3 -> 1e-4
    train_momentum = 0.9
    train_double_bias = True
    train_bias_decay = False
    lr_decay_step = 5

    # training
    epochs = 20

    pooling_size = 7

    # Anchor scales for RPN
    anchor_scales = [8, 16, 32]

    # Anchor ratios for RPN
    anchor_ratios = [0.5, 1, 2]

    # Feature stride for RPN
    feat_stride = 16

    pre_nms_topN = 12000
    post_nms_topN = 2000
    nms_thresh = 0.7
    rpn_min_size = 8

    rpn_clobber_positives = False
    rpn_negative_overlap = 0.3
    rpn_positive_overlap = 0.7
    rpn_fg_fraction = 0.25
    rpn_batchsize = 256
    rpn_bbox_inside_weights = (1.0, 1.0, 1.0, 1.0)
    rpn_positive_weight = -1.0

    fg_thresh = 0.5
    bg_thresh_hi = 0.5
    bg_thresh_lo = 0.0

    bbox_normalize_means = (0.0, 0.0, 0.0, 0.0)
    bbox_normalize_stds = (0.1, 0.1, 0.2, 0.2)

    bbox_normalize_targets_precomputed = True

    train_truncated = False

    pooling_mode = 'align'

    test_num = 10000

    # model
    load_path = None

    caffe_pretrain = False   # use caffe pretrained model instead of torchvision
    caffe_pretrain_path = 'checkpoints/vgg16_caffe.pth'

    def _parse(self, kwargs):
        state_dict = self._state_dict()
        for k, v in kwargs.items():
            if k not in state_dict:
                raise ValueError('UnKnown Option: "--%s"' % k)
            setattr(self, k, v)

        print('======user config========')
        pprint(self._state_dict())
        print('==========end============')

    def _state_dict(self):
        return {k: getattr(self, k) for k, _ in Config.__dict__.items() \
                if not k.startswith('_')}


opt = Config()

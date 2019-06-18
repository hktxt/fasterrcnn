from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import argparse
import time
import cv2
import torch
from utils.config import opt
from imageio import imread
from model.rpn.bbox_transform import clip_boxes
from utils.dataset import CusDataset
from torchvision.ops import nms
from model.rpn.bbox_transform import bbox_transform_inv
from model.utils.net_utils import save_net, load_net, vis_detections
from model.faster_rcnn.vgg16 import VGG16
from utils import torch_utils
# specify visible GPUs
os.environ["CUDA_VISIBLE_DEVICES"] = '0'


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--dataset', dest='dataset',
                        help='training dataset',
                        default='pascal_voc', type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='cfgs/vgg16.yml', type=str)
    parser.add_argument('--net', dest='net',
                        help='vgg16, res50, res101, res152',
                        default='vgg16', type=str)
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--load_dir', dest='load_dir',
                        help='directory to load models',
                        default="output")
    parser.add_argument('--image_dir', dest='image_dir',
                        help='directory to load images for demo',
                        default="samples")
    parser.add_argument('--save_dir', dest='save_dir',
                        help='directory to load images for demo',
                        default="images")
    parser.add_argument('--cuda', dest='cuda',
                        help='whether use CUDA',
                        action='store_true')
    parser.add_argument('--mGPUs', dest='mGPUs',
                        help='whether use multiple GPUs',
                        action='store_true')
    parser.add_argument('--cag', dest='class_agnostic',
                        help='whether perform class_agnostic bbox regression',
                        action='store_true')
    parser.add_argument('--parallel_type', dest='parallel_type',
                        help='which part of model to parallel, 0: all, 1: model before roi pooling',
                        default=0, type=int)
    parser.add_argument('--checkepoch', dest='checkepoch',
                        help='checkepoch to load network',
                        default=0, type=int)
    parser.add_argument('--checkpoint', dest='checkpoint',
                        help='checkpoint to load network',
                        default=5010, type=int)
    parser.add_argument('--bs', dest='batch_size',
                        help='batch_size',
                        default=1, type=int)
    parser.add_argument('--vis', dest='vis',
                        help='visualization mode',
                        action='store_true')
    parser.add_argument('--webcam_num', dest='webcam_num',
                        help='webcam ID number',
                        default=-1, type=int)

    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parse_args()
    device = torch_utils.select_device()
    print('Called with args:')
    print(args)


    input_dir = args.load_dir + "/" + args.net
    if not os.path.exists(input_dir):
        raise Exception('There is no input directory for loading network from ' + input_dir)
    load_name = os.path.join(input_dir, 'faster_rcnn_{}_{}.pth'.format(args.net, args.checkepoch))
    load_name = 'E:/condaDev/faster_rcnn_1_20_5010.pth'

    pascal_classes = np.asarray(['__background__',
                                 'aeroplane', 'bicycle', 'bird', 'boat',
                                 'bottle', 'bus', 'car', 'cat', 'chair',
                                 'cow', 'diningtable', 'dog', 'horse',
                                 'motorbike', 'person', 'pottedplant',
                                 'sheep', 'sofa', 'train', 'tvmonitor'])

    # initilize the network here.
    if args.net == 'vgg16':
        fasterRCNN = VGG16(pascal_classes, pretrained=False, class_agnostic=args.class_agnostic)
    elif args.net == 'res101':
        fasterRCNN = resnet(pascal_classes, 101, pretrained=False, class_agnostic=args.class_agnostic)
    elif args.net == 'res50':
        fasterRCNN = resnet(pascal_classes, 50, pretrained=False, class_agnostic=args.class_agnostic)
    elif args.net == 'res152':
        fasterRCNN = resnet(pascal_classes, 152, pretrained=False, class_agnostic=args.class_agnostic)
    else:
        print("network is not defined")

    fasterRCNN.create_architecture()

    print("load checkpoint {}".format(load_name))
    try:  # GPU
        checkpoint = torch.load(load_name)
    except:  # CPU
        checkpoint = torch.load(load_name, map_location=lambda storage, loc: storage)
    fasterRCNN.load_state_dict(checkpoint['model'])
    if 'pooling_mode' in checkpoint.keys():
        pooling_mode = checkpoint['pooling_mode']

    print('load model successfully!')

    data_set = CusDataset()
    data_loader = torch.utils.data.DataLoader(data_set, batch_size=1, num_workers=0)
    print('Loaded Photo: {} images.'.format(len(data_loader)))

    fasterRCNN.to(device)
    fasterRCNN.eval()

    start = time.time()
    max_per_image = 100
    thresh = 0.05

    for step, (im, im_data, im_info, gt_boxes, num_boxes, im_scales, pth) in enumerate(data_loader):
        im_data, im_info, gt_boxes, num_boxes, im_scales = im_data.to(device), im_info.to(device), \
                                                gt_boxes.to(device), num_boxes.to(device), im_scales.to(device)
        # pdb.set_trace()
        det_tic = time.time()

        rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_box, RCNN_loss_cls, RCNN_loss_bbox, rois_label\
            = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)

        scores = cls_prob.data
        boxes = rois.data[:, :, 1:5]

        # Apply bounding-box regression deltas
        box_deltas = bbox_pred.data
        box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(opt.bbox_normalize_stds).to(device) \
                     + torch.FloatTensor(opt.bbox_normalize_means).to(device)
        box_deltas = box_deltas.view(1, -1, 4 * len(pascal_classes))

        pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
        pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)

        pred_boxes /= im_scales[0]

        scores = scores.squeeze()
        pred_boxes = pred_boxes.squeeze()
        det_toc = time.time()
        detect_time = det_toc - det_tic
        misc_tic = time.time()
        im2show = np.copy(im)
        for j in range(1, len(pascal_classes)):
            inds = torch.nonzero(scores[:, j] > thresh).view(-1)
            # if there is det
            if inds.numel() > 0:
                cls_scores = scores[:, j][inds]
                _, order = torch.sort(cls_scores, 0, True)
                if args.class_agnostic:
                    cls_boxes = pred_boxes[inds, :]
                else:
                    cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]

                cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
                cls_dets = cls_dets[order]
                keep = nms(cls_boxes[order, :], cls_scores[order], opt.TEST_NMS)
                cls_dets = cls_dets[keep.view(-1).long()]
                im2show = vis_detections(im2show, pascal_classes[j], cls_dets.cpu().numpy(), 0.5)

        misc_toc = time.time()
        nms_time = misc_toc - misc_tic

        print('im_detect: {:d}/{:d} detect_time:{:.3f}s nms_time:{:.3f}s'
              .format(step + 1, len(data_loader), detect_time, nms_time))

        result_path = os.path.join(args.save_dir, pth[0])
        cv2.imwrite(result_path, im2show)
# very slow, ~10s per image
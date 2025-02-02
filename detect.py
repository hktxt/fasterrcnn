from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import numpy as np
import argparse
import time
import cv2
import torch
from utils.config import opt
from imageio import imread
from model.rpn.bbox_transform import clip_boxes
from utils.util import im_list_to_blob
from torchvision.ops import nms
from model.rpn.bbox_transform import bbox_transform_inv
from model.utils.net_utils import vis_detections_1
from model.faster_rcnn.vgg16 import VGG16
from model.faster_rcnn.resnet import resnet
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
    parser.add_argument('--checkepoch', dest='checkepoch',
                        help='checkepoch to load network',
                        default=19, type=int)
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


def _get_image_blob(im):
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
    blob = im_list_to_blob(processed_ims)

    return blob, np.array(im_scale_factors)


if __name__ == '__main__':

    args = parse_args()
    device = torch_utils.select_device()
    print('Called with args:')
    print(args)

    input_dir = args.load_dir + "/" + args.net
    #if not os.path.exists(input_dir):
        #raise Exception('There is no input directory for loading network from ' + input_dir)
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
        fasterRCNN = VGG16(pascal_classes, pretrained=False, class_agnostic=False)
    elif args.net == 'res101':
        fasterRCNN = resnet(pascal_classes, 101, pretrained=False, class_agnostic=False)
    elif args.net == 'res50':
        fasterRCNN = resnet(pascal_classes, 50, pretrained=False, class_agnostic=False)
    elif args.net == 'res152':
        fasterRCNN = resnet(pascal_classes, 152, pretrained=False, class_agnostic=False)
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

    # initilize the tensor holder here.
    im_data = torch.FloatTensor(1).to(device)
    im_info = torch.FloatTensor(1).to(device)
    num_boxes = torch.LongTensor(1).to(device)
    gt_boxes = torch.FloatTensor(1).to(device)

    with torch.no_grad():
        im_data = im_data
        im_info = im_info
        num_boxes = num_boxes
        gt_boxes = gt_boxes

    fasterRCNN.to(device)
    fasterRCNN.eval()

    start = time.time()
    max_per_image = 100
    thresh = 0.05
    vis = True
    webcam_num = args.webcam_num
    imglist = os.listdir(args.image_dir)
    num_images = len(imglist)
    print('Loaded Photo: {} images.'.format(num_images))

    while (num_images > 0):
        total_tic = time.time()
        if webcam_num == -1:
            num_images -= 1

        # Get image from the webcam
        if webcam_num >= 0:
            if not cap.isOpened():
                raise RuntimeError("Webcam could not open. Please check connection.")
            ret, frame = cap.read()
            im_in = np.array(frame)
        # Load the demo image
        else:
            im_file = os.path.join(args.image_dir, imglist[num_images])
            # im = cv2.imread(im_file)
            im_in = np.array(imread(im_file))
        if len(im_in.shape) == 2:
            im_in = im_in[:, :, np.newaxis]
            im_in = np.concatenate((im_in, im_in, im_in), axis=2)
        # rgb -> bgr
        im = im_in[:, :, ::-1]

        blobs, im_scales = _get_image_blob(im)
        assert len(im_scales) == 1, "Only single-image batch implemented"
        im_blob = blobs
        im_info_np = np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32)

        im_data_pt = torch.from_numpy(im_blob)
        im_data_pt = im_data_pt.permute(0, 3, 1, 2)
        im_info_pt = torch.from_numpy(im_info_np)

        im_data.resize_(im_data_pt.size()).copy_(im_data_pt)
        im_info.resize_(im_info_pt.size()).copy_(im_info_pt)
        gt_boxes.resize_(1, 1, 5).zero_()
        num_boxes.resize_(1).zero_()

        # pdb.set_trace()
        det_tic = time.time()

        with torch.no_grad():
            rois, cls_prob, bbox_pred, \
            rpn_loss_cls, rpn_loss_box, \
            RCNN_loss_cls, RCNN_loss_bbox, \
            rois_label = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)

        scores = cls_prob.data
        boxes = rois.data[:, :, 1:5]

        if opt.TEST_BBOX_REG:
            # Apply bounding-box regression deltas
            box_deltas = bbox_pred.data
            box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(opt.bbox_normalize_stds).to(device) \
                         + torch.FloatTensor(opt.bbox_normalize_means).to(device)
            box_deltas = box_deltas.view(1, -1, 4 * len(pascal_classes))

            pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
            pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
        else:
            # Simply repeat the boxes, once for each class
            pred_boxes = np.tile(boxes, (1, scores.shape[1]))

        pred_boxes /= im_scales[0]

        scores = scores.squeeze()
        pred_boxes = pred_boxes.squeeze()
        det_toc = time.time()
        detect_time = det_toc - det_tic
        misc_tic = time.time()
        if vis:
            im2show = np.copy(im)
        for j in range(1, len(pascal_classes)):
            inds = torch.nonzero(scores[:, j] > thresh).view(-1)
            # if there is det
            if inds.numel() > 0:
                cls_scores = scores[:, j][inds]
                _, order = torch.sort(cls_scores, 0, True)
                cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]

                cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
                cls_dets = cls_dets[order]
                keep = nms(cls_boxes[order, :], cls_scores[order], opt.TEST_NMS)
                cls_dets = cls_dets[keep.view(-1).long()]
                if vis:
                    im2show = vis_detections_1(im2show, pascal_classes[j], cls_dets.cpu().numpy(), 0.5)

        misc_toc = time.time()
        nms_time = misc_toc - misc_tic

        if webcam_num == -1:
            print('im_detect: {:d}/{:d} detect_time:{:.3f}s nms_time:{:.3f}s'
                  .format(num_images + 1, len(imglist), detect_time, nms_time))

        if vis and webcam_num == -1:
            # cv2.imshow('test', im2show)
            # cv2.waitKey(0)
            result_path = os.path.join(args.save_dir, imglist[num_images][:-4] + "_det.jpg")
            cv2.imwrite(result_path, im2show)
        else:
            im2showRGB = cv2.cvtColor(im2show, cv2.COLOR_BGR2RGB)
            cv2.imshow("frame", im2showRGB)
            total_toc = time.time()
            total_time = total_toc - total_tic
            frame_rate = 1 / total_time
            print('Frame rate:', frame_rate)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    if webcam_num >= 0:
        cap.release()
        cv2.destroyAllWindows()

# very slow, ~10s per image
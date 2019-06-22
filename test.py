import os
import torch
import time
import sys
import pickle
import argparse
import numpy as np
from tqdm import tqdm
from utils.config import opt
from utils.pascal_voc import evaluate_detections
from model.utils.net_utils import vis_detections
from torchvision.ops import nms
from utils.dataset import Dataset
from model.faster_rcnn.vgg16 import VGG16
from model.faster_rcnn.resnet import resnet
from model.rpn.bbox_transform import bbox_transform_inv, clip_boxes
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
    parser.add_argument('--split', dest='split',
                        help='splitting dataset',
                        default='test', type=str)
    parser.add_argument('--net', dest='net',
                        help='vgg16, res101',
                        default='vgg16', type=str)
    parser.add_argument('--nw', dest='num_workers',
                        help='number of worker to load data',
                        default=10, type=int)
    parser.add_argument('--load_dir', dest='load_dir',
                        help='directory to load models', default="output",
                        type=str)
    parser.add_argument('--cag', dest='class_agnostic',
                        help='whether perform class_agnostic bbox regression',
                        action='store_true')
    # config optimization
    parser.add_argument('--o', dest='optimizer',
                        help='training optimizer',
                        default="sgd", type=str)
    parser.add_argument('--lr', dest='lr',
                        help='starting learning rate',
                        default=0.001, type=float)
    parser.add_argument('--lr_decay_step', dest='lr_decay_step',
                        help='step to do learning rate decay, unit is epoch',
                        default=20, type=int)
    parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma',
                        help='learning rate decay ratio',
                        default=0.1, type=float)
    parser.add_argument('--nms_thresh', dest='nms_thresh',
                        help='threshold for nms',
                        default=0.3, type=float)

    # resume trained model
    parser.add_argument('--r', dest='resume',
                        help='resume checkpoint or not',
                        default=False, type=bool)
    parser.add_argument('--checkepoch', dest='checkepoch',
                        help='checkepoch to load model',
                        default=19, type=int)
    parser.add_argument('--vis', dest='vis',
                        help='visualization mode',
                        action='store_true')
    # log and diaplay
    parser.add_argument('--use_tb', dest='use_tfboard',
                        help='whether use tensorboard',
                        default=True, type=bool)

    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parse_args()
    print('Called with args:')
    print(args)

    device = torch_utils.select_device()
    torch.backends.cudnn.benchmark = True
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    #torch.cuda.empty_cache()

    print("Loading test data...")
    if args.dataset == "pascal_voc":
        dataset = Dataset(opt, split=args.split, filp=False)
        classes = dataset.db.label_names
        print("{} images were loaded from {}".format(len(dataset), opt.voc_data_dir))
        print("{} classes: {}".format(len(classes), classes))
    else:
        pass

    testloader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=args.num_workers, shuffle=False)
    model_dir = args.load_dir + "/" + args.net
    #if not os.path.exists(model_dir):
        #raise Exception('There is no input directory for loading network from: ' + model_dir)
    load_name = os.path.join(model_dir, 'faster_rcnn_{}_{}.pth'.format(args.net, args.checkepoch))
    load_name = os.path.join('E:/faster_rcnn_1_20_5010.pth')

    if args.net == 'vgg16':
        fasterRCNN = VGG16(classes, pretrained=False, class_agnostic=False)
    elif args.net == 'res50':
        fasterRCNN = resnet(classes, 50, pretrained=False, class_agnostic=False)
    elif args.net == 'res101':
        fasterRCNN = resnet(classes, 101, pretrained=False, class_agnostic=False)
    elif args.net == 'res152':
        fasterRCNN = resnet(classes, 152, pretrained=False, class_agnostic=False)
    else:
        raise Exception("network is not defined")
    print('creating {}...'.format(args.net))
    fasterRCNN.create_architecture()

    print("load checkpoint: {}".format(load_name))
    try:  # GPU
        checkpoint = torch.load(load_name)
    except:  # CPU
        checkpoint = torch.load(load_name, map_location=lambda storage, loc: storage)
    fasterRCNN.load_state_dict(checkpoint['model'])
    fasterRCNN.to(device)
    if 'pooling_mode' in checkpoint.keys():
        opt.pooling_mode = checkpoint['pooling_mode']
    print('load model successfully!')

    start = time.time()
    max_per_image = 100
    vis = args.vis
    if vis:
        thresh = 0.05
    else:
        thresh = 0.0

    save_name = 'faster_rcnn_10'
    num_images = len(dataset)
    all_boxes = [[[] for _ in range(num_images)] for _ in range(len(classes))]

    output_dir = 'result'

    fasterRCNN.eval()
    empty_array = np.transpose(np.array([[], [], [], [], []]), (1, 0))
    det_file = os.path.join(output_dir, 'detections.pkl')

    for i, (img, im_info, gt_boxes, num_boxes) in enumerate(tqdm(testloader)):
        img, im_info, gt_boxes, num_boxes = img.to(device), im_info.to(device), gt_boxes.to(device), num_boxes.to(
            device)

        with torch.no_grad():
            rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_box, RCNN_loss_cls, RCNN_loss_bbox, rois_label \
                = fasterRCNN(img, im_info, gt_boxes, num_boxes)

        det_tic = time.time()
        scores = cls_prob.data
        boxes = rois.data[:, :, 1:5]

        box_deltas = bbox_pred.data
        box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(opt.bbox_normalize_stds).to(device) \
                    + torch.FloatTensor(opt.bbox_normalize_means).to(device)
        box_deltas = box_deltas.view(1, -1, 4 * len(classes))

        pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
        pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)

        pred_boxes /= im_info[0][2].item()

        scores = scores.squeeze()
        pred_boxes = pred_boxes.squeeze()
        det_toc = time.time()
        detect_time = det_toc - det_tic
        misc_tic = time.time()

        for j in range(1, len(classes)):
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
                # cls_dets = torch.cat((cls_boxes, cls_scores), 1)
                cls_dets = cls_dets[order]
                keep = nms(cls_boxes[order, :], cls_scores[order], args.nms_thresh)
                cls_dets = cls_dets[keep.view(-1).long()]
                if vis:
                    im2show = vis_detections(im2show, classes[j], cls_dets.cpu().numpy(), 0.3)
                all_boxes[j][i] = cls_dets.cpu().numpy()
            else:
                all_boxes[j][i] = empty_array

        # Limit to max_per_image detections *over all classes*
        if max_per_image > 0:
            image_scores = np.hstack([all_boxes[j][i][:, -1] for j in range(1, len(classes))])
            if len(image_scores) > max_per_image:
                image_thresh = np.sort(image_scores)[-max_per_image]
                for j in range(1, len(classes)):
                    keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                    all_boxes[j][i] = all_boxes[j][i][keep, :]

        misc_toc = time.time()
        nms_time = misc_toc - misc_tic

        sys.stdout.write('im_detect: {:d}/{:d} {:.3f}s {:.3f}s   \r' \
                         .format(i + 1, num_images, detect_time, nms_time))
        sys.stdout.flush()

    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

    print('Evaluating detections')
    evaluate_detections(dataset.db.ids, classes, all_boxes, output_dir, args.split)



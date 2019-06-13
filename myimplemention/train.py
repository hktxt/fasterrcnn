import os
import torch
import time
import numpy as np
import torch.nn as nn
from utils.config import opt
from utils.dataset import Dataset, TestDataset, inverse_normalize
from model.faster_rcnn.vgg16 import VGG16
from model.utils.net_utils import adjust_learning_rate, clip_gradient
from utils import torch_utils
from tqdm import tqdm

if __name__ == '__main__':
    # specify visible GPUs
    os.environ["CUDA_VISIBLE_DEVICES"] = '2,3,4,5'
    device = torch_utils.select_device()

    torch.backends.cudnn.benchmark = True
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)

    print("Load data...")
    dataset = Dataset(opt)
    classes = dataset.db.label_names
    print("{} images were loaded from {}".format(len(dataset), opt.voc_data_dir))
    print("{} classes: {}".format(len(classes), classes))

    trainloader = torch.utils.data.DataLoader(dataset, batch_size=opt.bs, num_workers=opt.num_workers, shuffle=True)

    if opt.net == 'vgg16':
        fasterRCNN = VGG16(classes, pretrained=False, class_agnostic=False)
    fasterRCNN.create_architecture()

    lr = opt.lr
    params = []
    for key, value in dict(fasterRCNN.named_parameters()).items():
        if value.requires_grad:
            if 'bias' in key:
                params += [{'params': [value], 'lr': lr * (opt.train_double_bias + 1),
                            'weight_decay': opt.train_bias_decay and opt.weight_decay or 0}]
            else:
                params += [{'params': [value], 'lr': lr, 'weight_decay': opt.weight_decay}]

    if opt.optimizer == "adam":
        lr = lr * 0.1
        optimizer = torch.optim.Adam(params)
    elif opt.optimizer == "sgd":
        optimizer = torch.optim.SGD(params, momentum=opt.train_momentum)

    if torch.cuda.device_count() > 1:
        print("Let's use {} GPUs!".format(torch.cuda.device_count()))
        fasterRCNN = nn.DataParallel(fasterRCNN)
    fasterRCNN.to(device)

    for epoch in range(opt.epochs):
        fasterRCNN.train()
        loss_temp = 0
        start_time = time.time()

        if epoch % (opt.lr_decay_step + 1) == 0:
            adjust_learning_rate(optimizer, opt.lr_decay)
            lr *= opt.lr_decay

        for step, (img, im_info, gt_boxes, num_boxes) in enumerate(trainloader):
            img, im_info, gt_boxes, num_boxes = img.to(device), im_info.to(device), gt_boxes.to(device), num_boxes.to(
                device)
            fasterRCNN.zero_grad()  # optimizer.zero_grad() ?
            rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_box, RCNN_loss_cls, RCNN_loss_bbox, rois_label\
                = fasterRCNN(img, im_info, gt_boxes, num_boxes)

            loss = rpn_loss_cls.mean() + rpn_loss_box.mean() + RCNN_loss_cls.mean() + RCNN_loss_bbox.mean()
            loss_temp += loss.item()

            # backward
            optimizer.zero_grad()
            loss.backward()
            if opt.net == "vgg16":
                clip_gradient(fasterRCNN, 10.)
            optimizer.step()

            if step % opt.disp_interval == 0:
                end = time.time()
                if step > 0:
                    loss_temp /= (opt.disp_interval + 1)

                if torch.cuda.device_count() > 1:
                    loss_rpn_cls = rpn_loss_cls.mean().item()
                    loss_rpn_box = rpn_loss_box.mean().item()
                    loss_rcnn_cls = RCNN_loss_cls.mean().item()
                    loss_rcnn_box = RCNN_loss_bbox.mean().item()
                    fg_cnt = torch.sum(rois_label.data.ne(0))
                    bg_cnt = rois_label.data.numel() - fg_cnt
                else:
                    loss_rpn_cls = rpn_loss_cls.item()
                    loss_rpn_box = rpn_loss_box.item()
                    loss_rcnn_cls = RCNN_loss_cls.item()
                    loss_rcnn_box = RCNN_loss_bbox.item()
                    fg_cnt = torch.sum(rois_label.data.ne(0))
                    bg_cnt = rois_label.data.numel() - fg_cnt

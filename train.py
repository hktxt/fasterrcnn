import os
import time
import argparse
import numpy as np
# specify visible GPUs
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import torch
import torch.nn as nn
from utils.config import opt
from torch.utils.tensorboard import SummaryWriter
from utils.dataset import Dataset
from model.faster_rcnn.vgg16 import VGG16
from model.faster_rcnn.resnet import resnet
from evaluate import evaluate
from model.utils.net_utils import adjust_learning_rate, clip_gradient, save_checkpoint
from utils import torch_utils


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--dataset', dest='dataset',
                        help='training dataset',
                        default='pascal_voc', type=str)
    parser.add_argument('--train_split', dest='train_split',
                        help='splitting train set',
                        default='trainval', type=str)
    parser.add_argument('--test_split', dest='test_split',
                        help='splitting test set',
                        default='test', type=str)
    parser.add_argument('--net', dest='net',
                        help='vgg16, res101',
                        default='res50', type=str)
    parser.add_argument('--start_epoch', dest='start_epoch',
                        help='starting epoch',
                        default=0, type=int)
    parser.add_argument('--epochs', dest='max_epochs',
                        help='number of epochs to train',
                        default=20, type=int)
    parser.add_argument('--disp_interval', dest='disp_interval',
                        help='number of iterations to display',
                        default=5, type=int)
    parser.add_argument('--log_dir', dest='logs',
                        help='directory to save models', default="logs",
                        type=str)
    parser.add_argument('--save_dir', dest='save_dir',
                        help='directory to save models', default="output",
                        type=str)
    parser.add_argument('--nw', dest='num_workers',
                        help='number of worker to load data',
                        default=0, type=int)
    parser.add_argument('--bs', dest='batch_size',
                        help='batch_size',
                        default=1, type=int)

    # config optimization
    parser.add_argument('--o', dest='optimizer',
                        help='training optimizer',
                        default="sgd", type=str)
    parser.add_argument('--lr', dest='lr',
                        help='starting learning rate',
                        default=0.001, type=float)
    parser.add_argument('--lr_decay_step', dest='lr_decay_step',
                        help='step to do learning rate decay, unit is epoch',
                        default=5, type=int)
    parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma',
                        help='learning rate decay ratio',
                        default=0.1, type=float)

    # resume trained model
    parser.add_argument('--r', dest='resume',
                        help='resume checkpoint or not',
                        default=False, type=bool)
    parser.add_argument('--checkepoch', dest='checkepoch',
                        help='checkepoch to load model',
                        default=1, type=int)
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

    if args.use_tfboard:
        # using tensorboard
        print('using tensorboard for supervision...')
        writer = SummaryWriter(opt.logs)

    device = torch_utils.select_device()
    torch.backends.cudnn.benchmark = True
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)

    print("Loading train data...")
    if args.dataset == "pascal_voc":
        dataset = Dataset(opt, split=args.train_split)
        testset = Dataset(opt, split='test', filp=False)
        classes = dataset.db.label_names
        print("{} images were loaded from {}".format(len(dataset), opt.voc_data_dir))
        print("{} classes: {}".format(len(classes), classes))
    else:
        pass

    trainloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                              shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, num_workers=args.num_workers)

    # saving path
    output_dir = args.save_dir + "/" + args.net + "/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

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

    lr = args.lr
    params = []
    for key, value in dict(fasterRCNN.named_parameters()).items():
        if value.requires_grad:
            if 'bias' in key:
                params += [{'params': [value], 'lr': lr * (opt.train_double_bias + 1),
                            'weight_decay': opt.train_bias_decay and opt.weight_decay or 0}]
            else:
                params += [{'params': [value], 'lr': lr, 'weight_decay': opt.weight_decay}]

    if args.optimizer == "adam":
        lr = lr * 0.1
        optimizer = torch.optim.Adam(params)
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params, momentum=opt.train_momentum)

    if args.resume:
        load_name = os.path.join(output_dir, 'faster_rcnn_{}_{}.pth'.format(args.net, args.checkepoch))
        print("resuming training, loading checkpoint:{}".format(load_name))
        checkpoint = torch.load(load_name)
        args.start_epoch = checkpoint['epoch']
        fasterRCNN.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr = optimizer.param_groups[0]['lr']
        if 'pooling_mode' in checkpoint.keys():
            opt.pooling_mode = checkpoint['pooling_mode']
        print('load model successfully!')

    if torch.cuda.device_count() > 1:
        print("Let's use {} GPUs!".format(torch.cuda.device_count()))
        fasterRCNN = nn.DataParallel(fasterRCNN)
    fasterRCNN.to(device)

    print("start training...")
    begin = time.time()
    for epoch in range(args.start_epoch, args.max_epochs):
        fasterRCNN.train()
        loss_temp = 0
        start = time.time()

        if epoch % (args.lr_decay_step + 1) == 0:
            adjust_learning_rate(optimizer, args.lr_decay_gamma)
            lr *= args.lr_decay_gamma

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
            #if args.net == "vgg16":
            clip_gradient(fasterRCNN, 10.)
            optimizer.step()

            if step % args.disp_interval == 0:
                end = time.time()
                if step > 0:
                    loss_temp /= (args.disp_interval + 1)

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

                print("[Epoch:{}/{}]: iter:{}/{}, loss:{:.4f}, lr:{:.2e}"
                      .format(epoch, args.max_epochs, step, len(trainloader), loss_temp, lr))
                print("\t\t\tfg/bg={}/{}, time cost: {:.6f}s".format(fg_cnt, bg_cnt, end - start))
                print("\t\t\trpn_cls:{:.4f}, rpn_box:{:.4f}, rcnn_cls:{:.4f}, rcnn_box:{:.4f}"
                      .format(loss_rpn_cls, loss_rpn_box, loss_rcnn_cls, loss_rcnn_box))

                if args.use_tfboard:
                    info = {
                        'loss': loss_temp,
                        'loss_rpn_cls': loss_rpn_cls,
                        'loss_rpn_box': loss_rpn_box,
                        'loss_rcnn_cls': loss_rcnn_cls,
                        'loss_rcnn_box': loss_rcnn_box
                    }
                    writer.add_scalars("losses", info, (epoch - 1) * len(trainloader) + step)
                loss_temp = 0
                start = time.time()

        save_name = os.path.join(output_dir, 'faster_rcnn_{}_{}.pth'.format(args.net, epoch))
        save_checkpoint({
            'epoch': epoch + 1,
            'model': fasterRCNN.module.state_dict() if torch.cuda.device_count() > 1 else fasterRCNN.state_dict(),
            'optimizer': optimizer.state_dict(),
            'pooling_mode': opt.pooling_mode,
        }, save_name)
        print('save model:{}'.format(save_name))

        # eval
        #evaluate(testloader, fasterRCNN, 0.3, device)

    if args.use_tfboard:
        writer.close()

    finish = time.time()
    total_time = finish - begin
    seconds = total_time % 60
    minutes = total_time // 60
    hours = minutes // 60
    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(hours, minutes, seconds))

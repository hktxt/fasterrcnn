import os
import torch as t
from utils.config import opt
from model import FasterRCNNVGG16
from trainer import FasterRCNNTrainer
from data.util import  read_image
#from utils.vis_tool import vis_bbox
#from utils import array_tool as at
import cv2
import random
#%matplotlib inline


def draw_bbox(img, boxes, labels, scores, classes_name):
    assert len(boxes) == len(labels) == len(scores), "boxes, labels, scores should have the same length."
    N = len(boxes)
    tl = round(0.001 * max(img.shape[0:2])) + 1  # line thickness
    tf = max(tl - 1, 1)  # font thickness
    color = random.sample(range(0, 255), len(set(labels)))
    dict_label_color = dict(zip(set(labels), color))
    for box, label, score in zip(boxes, labels, scores):
        y_min, x_min, y_max, x_max = map(int, box)
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), dict_label_color[label], thickness=tl)
        info = classes_name[label] + " {:.2f}".format(score)
        t_size = cv2.getTextSize(info, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = x_min + t_size[0], y_min - t_size[1] - 3
        cv2.rectangle(img, (x_min, y_min), c2, dict_label_color[label], -1)  # filled

        cv2.putText(img, info, (x_min, y_min - 2), 0, tl / 3, [225, 255, 255],
                    thickness=tf, lineType=cv2.LINE_AA)
    #plt.imshow(img)
    #plt.show()
    return img


if __name__ == '__main__':

    VOC_BBOX_LABEL_NAMES = ['person']
    # img = read_image('samples/DepthFrame0070.png')

    faster_rcnn = FasterRCNNVGG16()
    trainer = FasterRCNNTrainer(faster_rcnn).cuda()

    # in this machine the cupy isn't install correctly...
    # so it's a little slow
    trainer.load('/data/hktxt/e/CV/fasterRCNN/simple-faster-rcnn-pytorch-depth/checkpoints/'
                 'fasterrcnn_06291454_0.9090404880754298')
    opt.caffe_pretrain=True # this model was trained from caffe-pretrained model
    label_names = VOC_BBOX_LABEL_NAMES + ['bg']

    test_pth = '/data/hktxt/e/CV/dataset/mianyang/xiaoningcaiji/1/'
    images = os.listdir(test_pth)
    images = [image for image in images if not image.endswith('.txt')]
    for i in range(len(images)):
        img_name = images[i]
        img_pth = os.path.join(test_pth, img_name)
        img = read_image(img_pth)
        img = t.from_numpy(img)[None]

        print("detecting: {}/{}".format(i+1, len(images)))
        _bboxes, _labels, _scores = trainer.faster_rcnn.predict(img, visualize=True)

        res_img = draw_bbox(img[0].permute(1,2,0).data.numpy(), _bboxes[0], _labels[0], _scores[0], label_names)
        cv2.imwrite(os.path.join('/data/hktxt/e/CV/fasterRCNN/simple-faster-rcnn-pytorch-depth/output/',img_name),
                    cv2.cvtColor(res_img, cv2.COLOR_RGB2BGR))

    # it failed to find the dog, but if you set threshold from 0.7 to 0.6, you'll find it
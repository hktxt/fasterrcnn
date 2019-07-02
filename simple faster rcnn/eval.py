from utils.config import opt
import os
import numpy as np
from model import FasterRCNNVGG16
from torch.utils import data as data_
from trainer import FasterRCNNTrainer
from data.util import read_image
from tqdm import tqdm
from data.dataset import EvalDataset
from data.dataset import preprocess
from utils.eval_tool import eval_detection_voc


class Dataset:
    def __init__(self, data_dir, use_difficult=False, return_difficult=False):
        self.data_dir = data_dir
        files = os.listdir(self.data_dir)
        ans = [file for file in files if file.endswith('.txt')]
        self.ans = [ano for ano in ans if os.path.getsize(os.path.join(self.data_dir, ano)) > 0]

        self.use_difficult = use_difficult
        self.return_difficult = return_difficult

    def __len__(self):
        return len(self.ans)

    def __getitem__(self, i):
        anno = self.ans[i]
        img_pth = os.path.join(self.data_dir, anno.replace('.txt', '.png'))
        anno_pth = os.path.join(self.data_dir, anno)

        ori_img = read_image(img_pth)  # BGR
        img = preprocess(ori_img)
        assert img is not None, 'File Not Found: {}'.format(img_pth)
        c, h, w = img.shape
        # load label
        bbox = []
        with open(anno_pth, 'r') as f:
            lines = f.read().splitlines()
        assert lines is not None, 'No annotations in: {}'.format(anno_pth)
        x = np.array([x.split() for x in lines], dtype=np.float32)
        if x.size > 0:
            # shit xywh to pixel xyxy
            bbox = x.copy()
            bbox[:, 1] = (x[:, 1] - x[:, 3] / 2)*w
            bbox[:, 2] = (x[:, 2] - x[:, 4] / 2)*h
            bbox[:, 3] = (x[:, 1] + x[:, 3] / 2)*w
            bbox[:, 4] = (x[:, 2] + x[:, 4] / 2)*h

        bboxes = bbox[:, 1:]
        bboxes = bboxes[:, [1, 0, 3, 2]]
        labels = bbox[:, 0].astype(np.int32)

        for b in bboxes:
            assert b[0] < b[2], print(b, anno_pth)
            assert b[1] < b[3], print(b, anno_pth)

        return img, ori_img.shape[1:], bboxes, labels

def make_files(pth):
    files = os.listdir(pth)
    files = [file for file in files if file.endswith('.txt')]
    with open(os.path.join(opt.voc_data_dir, 'eval.txt'), 'w') as f:
        for file in files:
            full_pth = os.path.join(pth, file)
            if os.path.getsize(full_pth) > 0:
                f.write(full_pth.replace('.txt', '.png') + '\n')


def eval(dataloader, faster_rcnn, test_num=10000):
    pred_bboxes, pred_labels, pred_scores = list(), list(), list()
    gt_bboxes, gt_labels, gt_difficults = list(), list(), list()
    for ii, (imgs, sizes, gt_bboxes_, gt_labels_) in tqdm(enumerate(dataloader)):
        sizes = [sizes[0][0].item(), sizes[1][0].item()]
        pred_bboxes_, pred_labels_, pred_scores_ = faster_rcnn.predict(imgs, [sizes])
        gt_bboxes += list(gt_bboxes_.numpy())
        gt_labels += list(gt_labels_.numpy())
        # gt_difficults += list(gt_difficults_.numpy())
        pred_bboxes += pred_bboxes_
        pred_labels += pred_labels_
        pred_scores += pred_scores_
        if ii == test_num: break

    result = eval_detection_voc(
        pred_bboxes, pred_labels, pred_scores,
        gt_bboxes, gt_labels,
        use_07_metric=True)
    return result


if __name__ == '__main__':

    test_pth = '/data/hktxt/e/CV/dataset/mianyang/xiaoningcaiji/1/'
    print("making {} file....".format(test_pth))
    make_files(test_pth)

    testset = EvalDataset(opt)
    test_dataloader = data_.DataLoader(testset,batch_size=1, num_workers=5, shuffle=False, pin_memory=True)

    #testset = Dataset(test_pth)
    #test_dataloader = data_.DataLoader(testset, batch_size=1, num_workers=5, shuffle=False, pin_memory=True)

    print("load {} testing images.".format(len(test_dataloader)))

    faster_rcnn = FasterRCNNVGG16()
    print('model construct completed')
    trainer = FasterRCNNTrainer(faster_rcnn).cuda()
    trainer.load('/data/hktxt/e/CV/fasterRCNN/simple-faster-rcnn-pytorch-depth/checkpoints/'
                 'fasterrcnn_06291454_0.9090404880754298')
    opt.caffe_pretrain = True  # this model was trained from caffe-pretrained model

    eval_result = eval(test_dataloader, faster_rcnn, test_num=10000)
    print(eval_result)
    # {'ap': array([0.66594607]), 'map': 0.6659460689498238}

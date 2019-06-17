import numpy as np
import torch
from torchvision.ops import nms
from model.rpn.bbox_transform import bbox_transform_inv, clip_boxes
from utils.config import opt
from utils.pascal_voc import evaluate_detections
from tqdm import tqdm


def evaluate(dataloader, model, nms_thresh, device):
    output_dir = opt.output_dir
    model.eval()
    classes = dataloader.dataset.db.label_names
    num_images = len(dataloader)
    print('evaluating {} images...'.format(num_images))
    max_per_image = 100
    all_boxes = [[[] for _ in range(num_images)] for _ in range(len(classes))]
    empty_array = np.transpose(np.array([[], [], [], [], []]), (1, 0))

    for i, (img, im_info, gt_boxes, num_boxes) in enumerate(tqdm(dataloader)):
        img, im_info, gt_boxes, num_boxes = img.to(device), im_info.to(device), gt_boxes.to(device), num_boxes.to(
            device)
        rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_box, RCNN_loss_cls, RCNN_loss_bbox, rois_label \
            = model(img, im_info, gt_boxes, num_boxes)

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

        for j in range(1, len(classes)):
            inds = torch.nonzero(scores[:, j] > 0.0).view(-1)
            # if there is det
            if inds.numel() > 0:
                cls_scores = scores[:, j][inds]
                _, order = torch.sort(cls_scores, 0, True)
                cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]

                cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
                # cls_dets = torch.cat((cls_boxes, cls_scores), 1)
                cls_dets = cls_dets[order]
                keep = nms(cls_boxes[order, :], cls_scores[order], nms_thresh)
                cls_dets = cls_dets[keep.view(-1).long()]
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

    model.train()
    print('Evaluating detections')
    evaluate_detections(dataloader.dataset.db.ids, classes, all_boxes, output_dir, 'test')
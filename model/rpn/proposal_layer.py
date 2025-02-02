from __future__ import absolute_import
# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------
# --------------------------------------------------------
# Reorganized and modified by Jianwei Yang and Jiasen Lu
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
from utils.config import opt
from .generate_anchors import generate_anchors
from .bbox_transform import bbox_transform_inv, clip_boxes, clip_boxes_batch
from torchvision.ops import nms

DEBUG = False


class _ProposalLayer(nn.Module):
    """
    Outputs object detection proposals by applying estimated bounding-box
    transformations to a set of regular boxes (called "anchors").
    """

    def __init__(self, feat_stride, scales, ratios):
        super(_ProposalLayer, self).__init__()

        self._feat_stride = feat_stride  # 16
        self._anchors = torch.from_numpy(generate_anchors(scales=np.array(scales),
            ratios=np.array(ratios))).float()     # torch.Size([9, 4])
        self._num_anchors = self._anchors.size(0)  # 9

        # rois blob: holds R regions of interest, each is a 5-tuple
        # (n, x1, y1, x2, y2) specifying an image batch index n and a
        # rectangle (x1, y1, x2, y2)
        # top[0].reshape(1, 5)
        #
        # # scores blob: holds scores for R regions of interest
        # if len(top) > 1:
        #     top[1].reshape(1, 1, 1, 1)

    def forward(self, input):

        # Algorithm:
        #
        # for each (H, W) location i
        #   generate A anchor boxes centered on cell i
        #   apply predicted bbox deltas at cell i to each of the A anchors
        # clip predicted boxes to image
        # remove predicted boxes with either height or width < threshold
        # sort all (proposal, score) pairs by score from highest to lowest
        # take top pre_nms_topN proposals before NMS
        # apply NMS with threshold 0.7 to remaining proposals
        # take after_nms_topN proposals after NMS
        # return the top proposals (-> RoIs top, scores top)

        # the first set of _num_anchors channels are bg probs
        # the second set are the fg probs

        # (rpn_cls_prob.data, rpn_bbox_pred.data, im_info, cfg_key)
        #        0                    1              2        3
        scores = input[0][:, self._num_anchors:, :, :]  # torch.Size([1, 9, 37, 56]) fg probs
        bbox_deltas = input[1]  # torch.Size([1, 36, 37, 56])
        im_info = input[2]      # tensor([[600.0000, 901.0000,   1.8018]], device='cuda:0')
        cfg_key = input[3]      # TRAIN

        pre_nms_topN  = opt.pre_nms_topN    # 12000
        post_nms_topN = opt.post_nms_topN   # 2000
        nms_thresh = opt.nms_thresh       # 0.7
        min_size = opt.rpn_min_size        # 8

        batch_size = bbox_deltas.size(0)

        feat_height, feat_width = scores.size(2), scores.size(3)  # 37, 56
        shift_x = np.arange(0, feat_width) * self._feat_stride
        """array([0, 16,  32,  48,  64,  80,  96,  112, 128, 144, 160, 176, 192,
                208, 224, 240, 256, 272, 288, 304, 320, 336, 352, 368, 384, 400,
                416, 432, 448, 464, 480, 496, 512, 528, 544, 560, 576, 592, 608,
                624, 640, 656, 672, 688, 704, 720, 736, 752, 768, 784, 800, 816,
                832, 848, 864, 880])"""
        shift_y = np.arange(0, feat_height) * self._feat_stride
        """array([  0,  16,  32,  48,  64,  80,  96, 112, 128, 144, 160, 176, 192,
                   208, 224, 240, 256, 272, 288, 304, 320, 336, 352, 368, 384, 400,
                   416, 432, 448, 464, 480, 496, 512, 528, 544, 560, 576])"""
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shifts = torch.from_numpy(np.vstack((shift_x.ravel(), shift_y.ravel(),
                                  shift_x.ravel(), shift_y.ravel())).transpose())  # torch.Size([2072, 4])
        shifts = shifts.contiguous().type_as(scores).float()

        A = self._num_anchors # 9
        K = shifts.size(0)  # 2072

        self._anchors = self._anchors.type_as(scores)
        # anchors = self._anchors.view(1, A, 4) + shifts.view(1, K, 4).permute(1, 0, 2).contiguous()
        anchors = self._anchors.view(1, A, 4) + shifts.view(K, 1, 4)      # torch.Size([2072, 9, 4])
        anchors = anchors.view(1, K * A, 4).expand(batch_size, K * A, 4)  # torch.Size([1, 18648, 4])

        # Transpose and reshape predicted bbox transformations to get them
        # into the same order as the anchors:

        bbox_deltas = bbox_deltas.permute(0, 2, 3, 1).contiguous()  # torch.Size([1, 37, 56, 36])
        bbox_deltas = bbox_deltas.view(batch_size, -1, 4)  # torch.Size([1, 18648, 4])

        # Same story for the scores:
        scores = scores.permute(0, 2, 3, 1).contiguous()  # torch.Size([1, 37, 56, 9])
        scores = scores.view(batch_size, -1)  # torch.Size([1, 18648])

        # Convert anchors into proposals via bbox transformations
        proposals = bbox_transform_inv(anchors, bbox_deltas, batch_size)  # torch.Size([1, 18648, 4])

        # 2. clip predicted boxes to image, less than 0 or larger than image.shape will be clamp
        proposals = clip_boxes(proposals, im_info, batch_size)  # torch.Size([1, 18648, 4])
        # proposals = clip_boxes_batch(proposals, im_info, batch_size)

        # assign the score to 0 if it's non keep.
        # keep = self._filter_boxes(proposals, min_size * im_info[:, 2])

        # trim keep index to make it euqal over batch
        # keep_idx = torch.cat(tuple(keep_idx), 0)

        # scores_keep = scores.view(-1)[keep_idx].view(batch_size, trim_size)
        # proposals_keep = proposals.view(-1, 4)[keep_idx, :].contiguous().view(batch_size, trim_size, 4)

        # _, order = torch.sort(scores_keep, 1, True)

        scores_keep = scores        # torch.Size([1, 18648])
        proposals_keep = proposals  # torch.Size([1, 18648, 4])
        _, order = torch.sort(scores_keep, 1, True)  # sort at dim 1, descending;  torch.Size([1, 18648])

        output = scores.new(batch_size, post_nms_topN, 5).zero_()  # torch.Size([1, 2000, 5])
        for i in range(batch_size):
            # # 3. remove predicted boxes with either height or width < threshold
            # # (NOTE: convert min_size to input image scale stored in im_info[2])
            proposals_single = proposals_keep[i]
            scores_single = scores_keep[i]

            # # 4. sort all (proposal, score) pairs by score from highest to lowest
            # # 5. take top pre_nms_topN (e.g. 6000)
            order_single = order[i]

            if 0 < pre_nms_topN < scores_keep.numel():  # numel: Calculate The Number Of Elements
                order_single = order_single[:pre_nms_topN]

            proposals_single = proposals_single[order_single, :]    # torch.Size([12000, 4])
            scores_single = scores_single[order_single].view(-1, 1)  # torch.Size([12000, 1])

            # 6. apply nms (e.g. threshold = 0.7)
            # 7. take after_nms_topN (e.g. 300)
            # 8. return the top proposals (-> RoIs top)

            # from model.roi_layers.nms import nms, numpy implementation
            # picked_boxes, picked_score = nms(proposals_single, scores_single.squeeze(1), nms_thresh)
            keep_idx_i = nms(proposals_single, scores_single.squeeze(1), nms_thresh)  # torch.Size([1947])
            keep_idx_i = keep_idx_i.long().view(-1)

            if post_nms_topN > 0:
                keep_idx_i = keep_idx_i[:post_nms_topN]  # torch.Size([2000])
            proposals_single = proposals_single[keep_idx_i, :]  # torch.Size([2000, 4])
            scores_single = scores_single[keep_idx_i, :]        # torch.Size([2000, 1])

            # padding 0 at the end.
            num_proposal = proposals_single.size(0)  # 2000
            output[i, :, 0] = i  # torch.Size([1, 2000, 5])
            output[i, :num_proposal, 1:] = proposals_single  # torch.Size([1, 2000, 5])

        return output

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

    def _filter_boxes(self, boxes, min_size):
        """Remove all boxes with any side smaller than min_size."""
        ws = boxes[:, :, 2] - boxes[:, :, 0] + 1
        hs = boxes[:, :, 3] - boxes[:, :, 1] + 1
        keep = ((ws >= min_size.view(-1, 1).expand_as(ws)) & (hs >= min_size.view(-1, 1).expand_as(hs)))
        return keep

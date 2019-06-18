import numpy as np


def nms(bounding_boxes, confidence_score, threshold):
    # If no bounding boxes, return empty list
    if len(bounding_boxes) == 0:
        return [], []

    # Bounding boxes
    boxes = np.array(bounding_boxes)
    '''[[187  82 337 317]
         [150  67 305 282]
         [246 121 368 304]]'''

    # coordinates of bounding boxes
    start_x = boxes[:, 0]
    start_y = boxes[:, 1]
    end_x = boxes[:, 2]
    end_y = boxes[:, 3]

    # Confidence scores of bounding boxes
    score = np.array(confidence_score)
    '[0.9  0.75 0.8 ]'

    # Picked bounding boxes
    picked_boxes = []
    picked_score = []

    # Compute areas of bounding boxes
    areas = (end_x - start_x + 1) * (end_y - start_y + 1)
    '[35636 33696 22632]'

    # Sort by confidence score of bounding boxes
    order = np.argsort(score)
    '[1 2 0]'

    # Iterate bounding boxes
    while order.size > 0:
        # The index of largest confidence score
        index = order[-1]  # 0,

        # Pick the bounding box with largest confidence score
        picked_boxes.append(bounding_boxes[index]) # [(187, 82, 337, 317)]
        picked_score.append(confidence_score[index]) # [0.9]
        # Compute ordinates of intersection-over-union(IOU)
        #print(end_y[order[:-1]])
        x1 = np.maximum(start_x[index], start_x[order[:-1]]) # 187,[150 246];
        x2 = np.minimum(end_x[index], end_x[order[:-1]]) # 337, [305 368];
        y1 = np.maximum(start_y[index], start_y[order[:-1]]) # 82, [ 67 121];
        y2 = np.minimum(end_y[index], end_y[order[:-1]]) # 317, [282 304];
        #print(x1,x2,y1,y2) [187 246] [305 337] [ 82 121] [282 304]

        # Compute areas of intersection-over-union
        w = np.maximum(0.0, x2 - x1 + 1) # [119.  92.]
        h = np.maximum(0.0, y2 - y1 + 1) # [201. 184.]
        intersection = w * h # [23919. 16928.]

        # Compute the ratio between intersection and union
        ratio = intersection / (areas[index] + areas[order[:-1]] - intersection) # [0.5266994  0.40948234]

        left = np.where(ratio < threshold) # (array([], dtype=int64),)
        order = order[left]

    return picked_boxes, picked_score

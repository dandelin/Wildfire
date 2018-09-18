from numba import jit
import numpy as np

@jit
def nms_kernel(boxes, thresh):
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)

    n_boxes = boxes.shape[0]
    suppressed = np.zeros((n_boxes, ), dtype=np.int)

    for i in range(n_boxes):
        if suppressed[i] == 1:
            continue
        ix1 = x1[i]
        iy1 = y1[i]
        ix2 = x2[i]
        iy2 = y2[i]
        iarea = areas[i]
        for j in range(i + 1, n_boxes):
            if suppressed[j] == 1:
                continue
            xx1 = max(ix1, x1[j])
            yy1 = max(iy1, y1[j])
            xx2 = min(ix2, x2[j])
            yy2 = min(iy2, y2[j])
            w = max(0.0, xx2 - xx1)
            h = max(0.0, yy2 - yy1)
            inter = w * h
            ovr = inter / (iarea + areas[j] - inter)
            if ovr >= thresh:
                suppressed[j] = 1

    return np.where(suppressed == 0)[0]

def nms(boxes, criteria, thresh, pre=None, post=None):
    """
    :param boxes: Tensor [N, 4], unsorted
    :param criteria: Tensor [N]
    :param thresh: Scalar
    :return:
    """

    crits, sort_idx = criteria.sort(descending=True)
    crits, sort_idx = crits[:pre], sort_idx[:pre]
    bbox = boxes[sort_idx]

    keep = nms_kernel(bbox.cpu().numpy(), thresh)
    keep = keep[:post]

    return sort_idx[keep]
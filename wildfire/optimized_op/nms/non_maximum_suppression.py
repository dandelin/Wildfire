from numba import cuda, jit
from numba.cuda.cudadrv import driver, devicearray, devices
import numpy as np
import math
import ctypes
import torch

@devices.require_context
def link_tensor(tensor):
    dtype = np.dtype(tensor.dtype.__str__().split('.')[1])
    ctx = driver.driver.get_context()
    mp = driver.MemoryPointer(ctx, ctypes.c_ulong(tensor.data_ptr()), tensor.numel() * tensor.element_size())
    da = devicearray.DeviceNDArray(tensor.size(), [i * tensor.element_size() for i in tensor.stride()], dtype,
                                   gpu_data=mp, stream=torch.cuda.current_stream().cuda_stream)
    return da

@cuda.jit(device=True)
def iou_device(box_a, box_b):
    top = max(box_a[0], box_b[0])
    bottom = min(box_a[2], box_b[2])
    left = max(box_a[1], box_b[1])
    right = min(box_a[3], box_b[3])
    intersection = (max(bottom - top, 0) * max(right - left, 0))
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    return intersection / (area_a + area_b - intersection)

@cuda.jit
def nms_kernel(bbox, ious, theta):
    x, y = cuda.grid(2)

    if x < ious.shape[0] and y < ious.shape[1]:
        ious[x, y] = 1 if iou_device(bbox[x], bbox[y]) > theta else 0

def nms(boxes, criteria, threshold):
    """
    :param boxes: Tensor [N, 4], unsorted
    :param criteria: Tensor [N]
    :param threshold: Scalar
    :return:
    """
    _, indices = criteria.sort(descending=True)
    n_box = boxes.size(0)
    thread_per_block = 16

    blocks = (math.ceil(n_box / thread_per_block), math.ceil(n_box / thread_per_block))
    threads = (thread_per_block, thread_per_block)

    sorted_boxes = boxes[indices]
    ious = torch.zeros(n_box, n_box).to(sorted_boxes)

    l_sorted_boxes = link_tensor(sorted_boxes)
    l_ious = link_tensor(ious)

    nms_kernel[blocks, threads](l_sorted_boxes, l_ious, threshold)


boxes = torch.tensor([
    [0, 0, 1, 1],
    [0, 0.1, 1, 1.1],
    [0, -0.1, 1, 0.9],
    [0, 10, 1, 11],
    [0, 10.1, 1, 11.1],
    [0, 100, 1, 101]
]).cuda()
criteria = torch.tensor([0.9, 0.75, 0.6, 0.95, 0.5, 0.3]).cuda()
thresh = 0.5

nms(boxes, criteria, thresh)
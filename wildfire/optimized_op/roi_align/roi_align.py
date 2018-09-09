import torch
from torch.autograd import Function
from numba import cuda, jit
from numba.cuda.cudadrv import driver, devicearray, devices
import numpy as np
import math
import ctypes
import torch

def link_tensor(tensor):
    dtype = np.dtype(tensor.dtype.__str__().split('.')[1])
    ctx = driver.driver.get_context()
    mp = driver.MemoryPointer(ctx, ctypes.c_ulong(tensor.data_ptr()), tensor.numel() * tensor.element_size())
    da = devicearray.DeviceNDArray(tensor.size(), [i * tensor.element_size() for i in tensor.stride()], dtype,
                                   gpu_data=mp, stream=torch.cuda.current_stream().cuda_stream)
    return da

@cuda.jit('void(float32[:],float32[:],float32[:],int32,int32,int32,int32,int32,int32,float32)')
def roi_align_forward_kernel(features, rois, output, f_c, f_h, f_w, r_n, aligned_height, aligned_width, spatial_scale):
    index = cuda.grid(1)

    if index >= r_n * f_c * aligned_width * aligned_height:
        return

    pw = index % aligned_width
    ph = (index // aligned_width) % aligned_height
    c = (index // aligned_width // aligned_height) % f_c
    n = index // aligned_width // aligned_height // f_c

    roi_batch_ind = rois[n * 5 + 0]
    roi_start_w = rois[n * 5 + 1] * spatial_scale
    roi_start_h = rois[n * 5 + 2] * spatial_scale
    roi_end_w = rois[n * 5 + 3] * spatial_scale
    roi_end_h = rois[n * 5 + 4] * spatial_scale

    roi_width = max(roi_end_w - roi_start_w, 0)
    roi_height = max(roi_end_h - roi_start_h, 0)
    bin_size_h = roi_height / aligned_height
    bin_size_w = roi_width / aligned_width

    h = ph * bin_size_h + roi_start_h
    w = pw * bin_size_w + roi_start_w

    hstart = min(h // 1, f_h)
    wstart = min(w // 1, f_w)

    feature_start = roi_batch_ind * f_c * f_h * f_w

    if h < 0 or h >= f_h or w < 0 or w >= f_w:
        output[index] = 0
    else:
        h_ratio = h - hstart
        w_ratio = w - wstart
        upleft = int(feature_start + (c * f_h + hstart) * f_w + wstart)
        upright = upleft + 1
        downleft = upleft + f_w
        downright = downleft + 1

        output[index] = \
            features[upleft] * (1 - h_ratio) * (1 - w_ratio) + \
            features[upright] * (1 - h_ratio) * w_ratio + \
            features[downleft] * h_ratio * (1 - w_ratio) + \
            features[downright] * h_ratio * w_ratio

@cuda.jit('void(float32[:],float32[:],float32[:],int32,int32,int32,int32,int32,int32,float32)')
def roi_align_backward_kernel(top_grad, rois, bottom_grad, f_c, f_w, f_h, r_n, aligned_height, aligned_width,
                              spatial_scale):
    index = cuda.grid(1)

    if index >= r_n * f_c * aligned_width * aligned_height:
        return

    pw = index % aligned_width
    ph = (index // aligned_width) % aligned_height
    c = (index // aligned_width // aligned_height) % f_c
    n = index // aligned_width // aligned_height // f_c

    roi_batch_ind = rois[n * 5 + 0]
    roi_start_w = rois[n * 5 + 1] * spatial_scale
    roi_start_h = rois[n * 5 + 2] * spatial_scale
    roi_end_w = rois[n * 5 + 3] * spatial_scale
    roi_end_h = rois[n * 5 + 4] * spatial_scale

    roi_width = max(roi_end_w - roi_start_w, 0)
    roi_height = max(roi_end_h - roi_start_h, 0)
    bin_size_h = roi_height / aligned_height
    bin_size_w = roi_width / aligned_width

    h = ph * bin_size_h + roi_start_h
    w = pw * bin_size_w + roi_start_w

    hstart = min(h // 1, f_h)
    wstart = min(w // 1, f_w)

    bottom_grad_start = roi_batch_ind * f_c * f_h * f_w

    if h >= 0 and h < f_h and w >= 0 and w < f_w:
        h_ratio = h - float(hstart)
        w_ratio = w - float(wstart)
        upleft = int(bottom_grad_start + (c * f_h + hstart) * f_w + wstart)
        upright = upleft + 1
        downleft = upleft + f_w
        downright = downleft + 1

        cuda.atomic.add(bottom_grad, upleft, top_grad[index] * (1 - h_ratio) * (1 - w_ratio))
        cuda.atomic.add(bottom_grad, upright, top_grad[index] * (1 - h_ratio) * w_ratio)
        cuda.atomic.add(bottom_grad, downleft, top_grad[index] * h_ratio * (1 - w_ratio))
        cuda.atomic.add(bottom_grad, downright, top_grad[index] * h_ratio * w_ratio)


def roi_align_forward_cuda(features, rois, aligned_height, aligned_width, spatial_scale):
    thread_per_block = 16
    output = features.new(rois.size(0), features.size(1), aligned_height, aligned_width).zero_()

    f_n, f_c, f_h, f_w = features.shape
    r_n, _5 = rois.shape

    features_flat = features.view(-1).contiguous()
    rois_flat = rois.view(-1).contiguous()
    output_flat = output.view(-1).contiguous()

    blocks = (math.ceil(output.numel() / thread_per_block),)
    threads = (thread_per_block,)

    features_l = link_tensor(features_flat)
    rois_l = link_tensor(rois_flat)
    output_l = link_tensor(output_flat)

    roi_align_forward_kernel[blocks, threads](
        features_l, rois_l, output_l, f_c, f_h, f_w, r_n, aligned_height, aligned_width, spatial_scale
    )
    return output_flat.view(rois.size(0), features.size(1), aligned_height, aligned_width)


def roi_align_backward_cuda(top_grad, rois, aligned_height, aligned_width, spatial_scale, feature_size):
    thread_per_block = 16
    f_n, f_c, f_h, f_w = feature_size
    bottom_grad = rois.new(f_n, f_c, f_h, f_w).zero_()

    r_n, _5 = rois.shape

    top_grad_flat = top_grad.view(-1).contiguous()
    rois_flat = rois.view(-1).contiguous()
    bottom_grad_flat = bottom_grad.view(-1).contiguous()

    blocks = (math.ceil(bottom_grad.numel() / thread_per_block),)
    threads = (thread_per_block,)

    top_grad_l = link_tensor(top_grad_flat)
    rois_l = link_tensor(rois_flat)
    bottom_grad_l = link_tensor(bottom_grad_flat)

    roi_align_backward_kernel[blocks, threads](
        top_grad_l, rois_l, bottom_grad_l, f_c, f_w, f_h, r_n, aligned_height, aligned_width, spatial_scale
    )

    return bottom_grad_flat.view(f_n, f_c, f_h, f_w)


class RoIAlignFunction(Function):
    def __init__(self, aligned_height, aligned_width, spatial_scale):
        self.aligned_width = aligned_width
        self.aligned_height = aligned_height
        self.spatial_scale = spatial_scale
        self.rois = None
        self.feature_size = None

    def forward(self, features, rois):
        self.rois = rois
        self.feature_size = features.size()

        output = roi_align_forward_cuda(features, rois, self.aligned_height, self.aligned_width, self.spatial_scale)
        return output

    def backward(self, top_grad):
        bottom_grad = roi_align_backward_cuda(top_grad, self.rois, self.aligned_height, self.aligned_width,
                                              self.spatial_scale, self.feature_size)

        return bottom_grad, None

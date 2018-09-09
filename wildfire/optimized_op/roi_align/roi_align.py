import torch
from torch.autograd import Function
from numba import cuda
import torch

from time import perf_counter as pc

def typestr(tensor):
    import sys
    endianness = '<' if sys.byteorder == 'little' else '>'
    types = {
        torch.float32: 'f4',
        torch.float: 'f4',
        torch.float64: 'f8',
        torch.double: 'f8',
        torch.float16: 'f2',
        torch.half: 'f2',
        torch.uint8: 'u1',
        torch.int8: 'i1',
        torch.int16: 'i2',
        torch.short: 'i2',
        torch.int32: 'i4',
        torch.int: 'i4',
        torch.int64: 'i8',
        torch.long: 'i8'
    }
    return endianness + types[tensor.dtype]

def link_tensor(tensor):
    cai_dict = {
        'shape': tuple(tensor.shape),
        'data': (tensor.data_ptr(), False),
        'typestr': typestr(tensor),
        'version': 0,
        'strides': [i * tensor.element_size() for i in tensor.stride()],
        'descr': [('', typestr(tensor))]
    }
    setattr(tensor, '__cuda_array_interface__', cai_dict)
    device_array = cuda.as_cuda_array(tensor)
    return device_array

@cuda.jit
def roi_align_forward_kernel(features, rois, output, f_c, f_h, f_w, r_n, aligned_height, aligned_width, spatial_scale):
    index = cuda.grid(1)

    if index < r_n * f_c * aligned_width * aligned_height:

        pw = index % aligned_width
        ph = (index // aligned_width) % aligned_height
        c = (index // aligned_width // aligned_height) % f_c
        n = index // aligned_width // aligned_height // f_c

        roi_batch_ind = rois[n * 5 + 0]
        roi_start_w = rois[n * 5 + 1] * spatial_scale
        roi_start_h = rois[n * 5 + 2] * spatial_scale
        roi_end_w = rois[n * 5 + 3] * spatial_scale
        roi_end_h = rois[n * 5 + 4] * spatial_scale

        roi_width = max(roi_end_w - roi_start_w, 0.)
        roi_height = max(roi_end_h - roi_start_h, 0.)
        bin_size_h = roi_height / aligned_height
        bin_size_w = roi_width / aligned_width

        h = ph * bin_size_h + roi_start_h
        w = pw * bin_size_w + roi_start_w

        hstart = min(h // 1, f_h)
        wstart = min(w // 1, f_w)

        feature_start = roi_batch_ind * f_c * f_h * f_w

        interpolated = 0

        if h < 0 or h >= f_h or w < 0 or w >= f_w:
            output[index] = 0
        else:
            h_ratio = h - hstart
            w_ratio = w - wstart
            upleft = int(feature_start + (c * f_h + hstart) * f_w + wstart)
            upright = upleft + 1
            downleft = upleft + f_w
            downright = downleft + 1

            interpolated += features[upleft] * (1 - h_ratio) * (1 - w_ratio)
            interpolated += features[upright] * (1 - h_ratio) * w_ratio
            interpolated += features[downleft] * h_ratio * (1 - w_ratio)
            interpolated += features[downright] * h_ratio * w_ratio
            output[index] = interpolated


@cuda.jit
def roi_align_backward_kernel(top_grad, rois, bottom_grad, f_c, f_w, f_h, r_n, aligned_height, aligned_width,
                              spatial_scale):
    index = cuda.grid(1)

    if index < r_n * f_c * aligned_width * aligned_height:

        pw = index % aligned_width
        ph = (index // aligned_width) % aligned_height
        c = (index // aligned_width // aligned_height) % f_c
        n = index // aligned_width // aligned_height // f_c

        roi_batch_ind = rois[n * 5 + 0]
        roi_start_w = rois[n * 5 + 1] * spatial_scale
        roi_start_h = rois[n * 5 + 2] * spatial_scale
        roi_end_w = rois[n * 5 + 3] * spatial_scale
        roi_end_h = rois[n * 5 + 4] * spatial_scale

        roi_width = max(roi_end_w - roi_start_w, 0.)
        roi_height = max(roi_end_h - roi_start_h, 0.)
        bin_size_h = roi_height / aligned_height
        bin_size_w = roi_width / aligned_width

        h = ph * bin_size_h + roi_start_h
        w = pw * bin_size_w + roi_start_w

        hstart = min(h // 1, f_h)
        wstart = min(w // 1, f_w)

        bottom_grad_start = roi_batch_ind * f_c * f_h * f_w

        if not (h < 0 or h >= f_h or w < 0 or w >= f_w):
            h_ratio = h - hstart
            w_ratio = w - wstart
            upleft = int(bottom_grad_start + (c * f_h + hstart) * f_w + wstart)
            upright = upleft + 1
            downleft = upleft + f_w
            downright = downleft + 1

            cuda.atomic.add(bottom_grad, upleft, top_grad[index] * (1. - h_ratio) * (1. - w_ratio))
            cuda.atomic.add(bottom_grad, upright, top_grad[index] * (1. - h_ratio) * w_ratio)
            cuda.atomic.add(bottom_grad, downleft, top_grad[index] * h_ratio * (1. - w_ratio))
            cuda.atomic.add(bottom_grad, downright, top_grad[index] * h_ratio * w_ratio)


def roi_align_forward_cuda(features, rois, aligned_height, aligned_width, spatial_scale):
    # tic = pc()
    thread_per_block = 1024
    output = features.new(rois.size(0), features.size(1), aligned_height, aligned_width).zero_()

    f_n, f_c, f_h, f_w = features.shape
    r_n, _5 = rois.shape

    features_flat = features.reshape(-1)
    rois_flat = rois.reshape(-1)
    output_flat = output.reshape(-1)

    blocks = (output.numel() + thread_per_block - 1) // thread_per_block
    threads = thread_per_block

    features_l = link_tensor(features_flat)
    rois_l = link_tensor(rois_flat)
    output_l = link_tensor(output_flat)
    # torch.cuda.synchronize()
    # toc = pc()
    # print((toc - tic) * 1e3)

    # tic = pc()
    roi_align_forward_kernel[blocks, threads](
        features_l, rois_l, output_l, f_c, f_h, f_w, r_n, aligned_height, aligned_width, spatial_scale
    )
    # torch.cuda.synchronize()
    # toc = pc()
    # print((toc - tic) * 1e3)

    return output_flat.reshape(rois.size(0), features.size(1), aligned_height, aligned_width)


def roi_align_backward_cuda(top_grad, rois, aligned_height, aligned_width, spatial_scale, feature_size):
    thread_per_block = 1024
    f_n, f_c, f_h, f_w = feature_size
    bottom_grad = top_grad.new(f_n, f_c, f_h, f_w).zero_()

    r_n, _5 = rois.shape

    top_grad_flat = top_grad.reshape(-1)
    rois_flat = rois.reshape(-1)
    bottom_grad_flat = bottom_grad.reshape(-1)

    blocks = (r_n * aligned_height * aligned_width * f_c + thread_per_block - 1) // thread_per_block
    threads = thread_per_block

    top_grad_l = link_tensor(top_grad_flat)
    rois_l = link_tensor(rois_flat)
    bottom_grad_l = link_tensor(bottom_grad_flat)

    roi_align_backward_kernel[blocks, threads](
        top_grad_l, rois_l, bottom_grad_l, f_c, f_w, f_h, r_n, aligned_height, aligned_width, spatial_scale
    )

    return bottom_grad_flat.reshape(f_n, f_c, f_h, f_w)


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

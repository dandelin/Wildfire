import torch
from torch.autograd import Function
from numba import cuda
import torch
import math

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
def roi_align_forward_kernel(features, rois, bids, output, f_c, f_h, f_w, r_n, aligned_height, aligned_width):
    index = cuda.grid(1)

    if index < r_n * f_c * aligned_width * aligned_height:

        pw = index % aligned_width
        ph = index // aligned_width % aligned_height
        c = index // aligned_width // aligned_height % f_c
        n = index // aligned_width // aligned_height // f_c

        roi_start_w, roi_start_h, bin_size_w, bin_size_h = rois[n * 4: n * 4 + 4]

        h = ph * bin_size_h + roi_start_h
        w = pw * bin_size_w + roi_start_w

        if h < 0 or h >= f_h or w < 0 or w >= f_w:
            output[index] = 0
        else:
            hstart = min(math.floor(h), f_h)
            wstart = min(math.floor(w), f_w)

            h_ratio = h - hstart
            w_ratio = w - wstart
            upleft = int(bids[n] + c * f_h * f_w + hstart * f_w + wstart)

            output[index] = features[upleft] * (1 - h_ratio) * (1 - w_ratio) + \
                            features[upleft + 1] * (1 - h_ratio) * w_ratio + \
                            features[upleft + f_w] * h_ratio * (1 - w_ratio) + \
                            features[upleft + f_w + 1] * h_ratio * w_ratio


@cuda.jit
def roi_align_backward_kernel(top_grad, rois, bids, bottom_grad, f_c, f_w, f_h, r_n, aligned_height, aligned_width):
    index = cuda.grid(1)

    if index < r_n * f_c * aligned_width * aligned_height:

        pw = index % aligned_width
        ph = index // aligned_width % aligned_height
        c = index // aligned_width // aligned_height % f_c
        n = index // aligned_width // aligned_height // f_c

        roi_start_w, roi_start_h, bin_size_w, bin_size_h = rois[n * 4: n * 4 + 4]

        h = ph * bin_size_h + roi_start_h
        w = pw * bin_size_w + roi_start_w

        if not (h < 0 or h >= f_h or w < 0 or w >= f_w):
            hstart = min(math.floor(h), f_h)
            wstart = min(math.floor(w), f_w)

            h_ratio = h - hstart
            w_ratio = w - wstart
            upleft = int(bids[n] + c * f_h * f_w + hstart * f_w + wstart)

            cuda.atomic.add(bottom_grad, upleft, top_grad[index] * (1. - h_ratio) * (1. - w_ratio))
            cuda.atomic.add(bottom_grad, upleft + 1, top_grad[index] * (1. - h_ratio) * w_ratio)
            cuda.atomic.add(bottom_grad, upleft + f_w, top_grad[index] * h_ratio * (1. - w_ratio))
            cuda.atomic.add(bottom_grad, upleft + f_w + 1, top_grad[index] * h_ratio * w_ratio)


def roi_align_forward_cuda(features, rois, bids, aligned_height, aligned_width, spatial_scale):
    thread_per_block = 64
    f_n, f_c, f_h, f_w = features.shape
    r_n, _5 = rois.shape

    output = features.new(rois.size(0), features.size(1), aligned_height, aligned_width).zero_()

    new_bids = bids * f_c * f_h * f_w
    new_rois = rois * spatial_scale

    roi_width = torch.clamp(new_rois[:, 2] - new_rois[:, 0], 0)
    roi_height = torch.clamp(new_rois[:, 3] - new_rois[:, 1], 0)
    roi_bin_width = roi_width / aligned_width
    roi_bin_height = roi_height / aligned_height

    new_rois[:, 2] = roi_bin_width
    new_rois[:, 3] = roi_bin_height

    features_flat = features.reshape(-1)
    rois_flat = new_rois.reshape(-1)
    bids_flat = new_bids.reshape(-1)
    output_flat = output.reshape(-1)

    blocks = (output.numel() + thread_per_block - 1) // thread_per_block
    threads = thread_per_block

    features_l = link_tensor(features_flat)
    rois_l = link_tensor(rois_flat)
    bids_l = link_tensor(bids_flat)
    output_l = link_tensor(output_flat)

    roi_align_forward_kernel[blocks, threads](
        features_l, rois_l, bids_l, output_l, f_c, f_h, f_w, r_n, aligned_height, aligned_width
    )
    with open('inspection_forward.txt', 'w') as fp:
        roi_align_forward_kernel.inspect_types(file=fp)

    return output_flat.reshape(rois.size(0), features.size(1), aligned_height, aligned_width)


def roi_align_backward_cuda(top_grad, rois, bids, aligned_height, aligned_width, spatial_scale, feature_size):
    thread_per_block = 64
    f_n, f_c, f_h, f_w = feature_size
    r_n, _5 = rois.shape

    bottom_grad = top_grad.new(f_n, f_c, f_h, f_w).zero_()

    new_bids = bids * f_c * f_h * f_w
    new_rois = rois * spatial_scale

    roi_width = torch.clamp(new_rois[:, 2] - new_rois[:, 0], 0)
    roi_height = torch.clamp(new_rois[:, 3] - new_rois[:, 1], 0)
    roi_bin_width = roi_width / aligned_width
    roi_bin_height = roi_height / aligned_height

    new_rois[:, 2] = roi_bin_width
    new_rois[:, 3] = roi_bin_height

    top_grad_flat = top_grad.reshape(-1)
    rois_flat = new_rois.reshape(-1)
    bids_flat = new_bids.reshape(-1)
    bottom_grad_flat = bottom_grad.reshape(-1)

    blocks = (r_n * aligned_height * aligned_width * f_c + thread_per_block - 1) // thread_per_block
    threads = thread_per_block

    top_grad_l = link_tensor(top_grad_flat)
    rois_l = link_tensor(rois_flat)
    bids_l = link_tensor(bids_flat)
    bottom_grad_l = link_tensor(bottom_grad_flat)

    roi_align_backward_kernel[blocks, threads](
        top_grad_l, rois_l, bids_l, bottom_grad_l, f_c, f_w, f_h, r_n, aligned_height, aligned_width
    )
    # with open('inspection_backward.txt', 'w') as fp:
    #     roi_align_backward_kernel.inspect_types(file=fp)

    return bottom_grad_flat.reshape(f_n, f_c, f_h, f_w)


class RoIAlignFunction(Function):
    def __init__(self, aligned_height, aligned_width, spatial_scale):
        self.aligned_width = aligned_width
        self.aligned_height = aligned_height
        self.spatial_scale = spatial_scale
        self.rois = None
        self.bids = None
        self.feature_size = None

    def forward(self, features, rois, bids):
        self.rois = rois
        self.bids = bids
        self.feature_size = features.size()

        output = roi_align_forward_cuda(features, rois, bids, self.aligned_height, self.aligned_width, self.spatial_scale)
        return output

    def backward(self, top_grad):
        bottom_grad = roi_align_backward_cuda(top_grad, self.rois, self.bids, self.aligned_height, self.aligned_width,
                                              self.spatial_scale, self.feature_size)

        return bottom_grad, None, None

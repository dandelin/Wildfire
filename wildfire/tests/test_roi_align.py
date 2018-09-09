from wildfire.optimized_op.roi_align.roi_align_native.functions.roi_align import RoIAlignFunction as roi_align_native
from wildfire.optimized_op.roi_align.roi_align import RoIAlignFunction as roi_align_numba
from time import perf_counter as pc
import torch
import tqdm

def test():
    rand_xy = torch.rand(10 ** 4, 2).cuda()
    boxes = torch.cat([rand_xy, rand_xy + 10], dim=1)
    batch_idx = (torch.rand(10 ** 4, 1) > 0.5).float().cuda()
    rois = torch.cat([batch_idx, boxes], dim=1)

    features1 = torch.rand(2, 1024, 5, 5).cuda().requires_grad_()
    features2 = features1.clone().detach().requires_grad_()

    numba_time_forward = []
    numba_time_backward = []
    native_time_forward = []
    native_time_backward = []

    progress = tqdm.tqdm(range(10 ** 1))

    for i in progress:
        if features1.grad is not None:
            features1.grad.zero_()
        if features2.grad is not None:
            features2.grad.zero_()

        tic = pc()
        pooled1 = roi_align_numba(7, 7, 1)(features1, rois)
        toc = pc()
        numba_time_forward.append(toc - tic)

        tic = pc()
        pooled1.mean().backward()
        toc = pc()
        numba_time_backward.append(toc - tic)

        tic = pc()
        pooled2 = roi_align_native(7, 7, 1)(features2, rois)
        toc = pc()
        native_time_forward.append(toc - tic)

        tic = pc()
        pooled2.mean().backward()
        toc = pc()
        native_time_backward.append(toc - tic)

        if i == 0:
            print(pooled1[0, 0])
            print(pooled2[0, 0])
            print(features1.grad[0, 0])
            print(features2.grad[0, 0])

        progress.set_description(f'Error forward : {(pooled1.float() - pooled2.float()).mean():.4f} '
                                 f'Error backward : {(features1.grad - features2.grad).mean():.4f}')

    print(f'numba_roi_align_forward : min {min(numba_time_forward[1:]) * 1e+6 / 10**3:.4f}us\n'
          f'max {max(numba_time_forward[1:]) * 1e+6 / 10**3:.4f}us\n'
          f'sum {sum(numba_time_forward[1:]) * 1e+6 / 10**3:.4f}us\n'
          f'numba_roi_align_backward : min {min(numba_time_backward[1:]) * 1e+6 / 10**3:.4f}us\n'
          f'max {max(numba_time_backward[1:]) * 1e+6 / 10**3:.4f}us\n'
          f'sum {sum(numba_time_backward[1:]) * 1e+6 / 10**3:.4f}us\n'
          f'original_roi_align_forward : min {min(native_time_forward[1:]) * 1e+6 / 10**3:.4f}us\n'
          f'max {max(native_time_forward[1:]) * 1e+6 / 10**3:.4f}us\n'
          f'sum {sum(native_time_forward[1:]) * 1e+6 / 10**3:.4f}us\n'
          f'original_roi_align_backward : min {min(native_time_backward[1:]) * 1e+6 / 10**3:.4f}us\n'
          f'max {max(native_time_backward[1:]) * 1e+6 / 10**3:.4f}us\n'
          f'sum {sum(native_time_backward[1:]) * 1e+6 / 10**3:.4f}us')

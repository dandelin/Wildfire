from wildfire.optimized_ops.roi_align.roi_align_native.modules.roi_align import RoIAlignAvg as roi_align_native
from wildfire.optimized_ops.roi_align.roi_align import RoIAlignAvg as roi_align_numba
from time import perf_counter as pc
from numba import cuda
import torch
import tqdm

def test():
    N = 10 ** 3

    rand_xy = torch.rand(2 * 512, 2).cuda()  # two image per GPU, 512 RoIs sampled
    boxes = torch.cat([rand_xy + 1, rand_xy + 4], dim=1)

    rois1 = boxes.clone()
    bids1 = (torch.rand(2 * 512) > 0.5).to(boxes).long()

    rois2 = torch.cat([bids1.clone().unsqueeze(1).float(), rois1.clone()], dim=1)

    features1 = torch.rand(2, 256, 20, 20).cuda()
    features2 = features1.clone()

    features1.requires_grad_()
    features2.requires_grad_()

    numba_time_forward = []
    numba_time_backward = []
    native_time_forward = []
    native_time_backward = []

    progress = tqdm.tqdm(range(N))

    torch.cuda.synchronize()
    torch.cuda.synchronize()

    for i in progress:
        if features1.grad is not None:
            features1.grad.zero_()
        if features2.grad is not None:
            features2.grad.zero_()

        tic = pc()
        pooled1 = roi_align_numba(7, 7, 1)(features1, rois1, bids1)
        torch.cuda.synchronize()
        toc = pc()
        numba_time_forward.append(toc - tic)

        tic = pc()
        pooled1.mean().backward()
        torch.cuda.synchronize()
        toc = pc()
        numba_time_backward.append(toc - tic)

        tic = pc()
        pooled2 = roi_align_native(7, 7, 1)(features2, rois2)
        torch.cuda.synchronize()
        toc = pc()
        native_time_forward.append(toc - tic)

        tic = pc()
        pooled2.mean().backward()
        torch.cuda.synchronize()
        toc = pc()
        native_time_backward.append(toc - tic)

        # if i == 0:
        #     ne = (torch.abs(pooled1 - pooled2) > 1e-2)
        #     print(pooled1[ne])
        #     print(pooled2[ne])
        #     ne = (torch.abs(features1.grad - features2.grad) > 1e-2)
        #     print(features1.grad[ne])
        #     print(features2.grad[ne])

        progress.set_description(f'Error forward : {(pooled1 - pooled2).mean():.4f} '
                                 f'Error backward : {(features1.grad - features2.grad).mean():.4f}')

    print(f'numba_roi_align_forward : min {min(numba_time_forward[1:]) * 1e+3:.4f}ms '
          f'max {max(numba_time_forward[1:]) * 1e+3:.4f}ms '
          f'mean {sum(numba_time_forward[1:]) * 1e+3 / (N - 1):.4f}ms\n'
          f'numba_roi_align_backward : min {min(numba_time_backward[1:]) * 1e+3:.4f}ms '
          f'max {max(numba_time_backward[1:]) * 1e+3:.4f}ms '
          f'mean {sum(numba_time_backward[1:]) * 1e+3 / (N - 1):.4f}ms\n'
          f'original_roi_align_forward : min {min(native_time_forward[1:]) * 1e+3:.4f}ms '
          f'max {max(native_time_forward[1:]) * 1e+3:.4f}ms '
          f'mean {sum(native_time_forward[1:]) * 1e+3 / (N - 1):.4f}ms\n'
          f'original_roi_align_backward : min {min(native_time_backward[1:]) * 1e+3:.4f}ms '
          f'max {max(native_time_backward[1:]) * 1e+3:.4f}ms '
          f'mean {sum(native_time_backward[1:]) * 1e+3 / (N - 1):.4f}ms')

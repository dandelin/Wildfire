from wildfire.optimized_ops.nms.nms_native.nms_wrapper import nms as nms_native_kernel
from wildfire.optimized_ops.nms.nms import nms as nms_numba
from time import perf_counter as pc
import torch
import tqdm

def nms_native(boxes, crits, overlap=0.5):
    crits, sort_idx = crits.sort(descending=True)
    bbox = boxes[sort_idx]
    dets = torch.cat([bbox, crits.unsqueeze(1)], dim=1)
    keep = nms_native_kernel(dets, overlap).long().reshape(-1)

    return sort_idx[keep]


def test():
    N = 10 ** 3

    rand_xy = torch.rand(10 ** 4, 2).cuda()
    boxes = torch.cat([rand_xy, rand_xy + 2], dim=1)
    criteria = torch.rand(10 ** 4).cuda()
    thresh = 0.5

    numba_time = []
    native_time = []

    progress = tqdm.tqdm(range(N))

    for _ in progress:
        tic = pc()
        keep1 = nms_numba(boxes, criteria, thresh)
        toc = pc()
        numba_time.append(toc - tic)

        tic = pc()
        keep2 = nms_native(boxes, criteria, thresh)
        torch.cuda.synchronize()
        toc = pc()
        native_time.append(toc - tic)

        progress.set_description(f'Error : {(keep1.float() - keep2.float()).mean():.4f}')

    print(f'numba_nms : min {min(numba_time[1:]) * 1e+3:.4f}ms '
          f'max {max(numba_time[1:]) * 1e+3:.4f}ms '
          f'mean {sum(numba_time[1:]) * 1e+3 / (N - 1):.4f}ms\n'
          f'cuda_nms : min {min(native_time[1:]) * 1e+3:.4f}ms '
          f'max {max(native_time[1:]) * 1e+3:.4f}ms '
          f'mean {sum(native_time[1:]) * 1e+3 / (N - 1):.4f}ma')

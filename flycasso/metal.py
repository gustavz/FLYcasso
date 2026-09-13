"""Full-graph CSR multiply on Apple GPUs, with the transposed graph for backward.

PyTorch MPS cannot store CSR tensors; keep their three arrays in ordinary Metal
buffers. No edge pruning, dense N×N matrix, CPU fallback or reduced precision.
"""
from functools import lru_cache

import torch


@lru_cache(maxsize=1)
def kernels():
    return torch.mps.compile_shader(r'''
#include <metal_stdlib>
using namespace metal;
kernel void csr_mm(device const int* ptr, device const int* idx,
                   device const float* val, device const float* x,
                   device float* out, constant uint& columns, constant uint& width,
                   uint2 tid [[thread_position_in_grid]],
                   uint lane [[thread_index_in_simdgroup]]) {
    // Adjacent lanes load adjacent batch columns; remaining lanes share edges.
    uint row = tid.x / 32, col = tid.y * width + lane % width;
    float sum = 0;
    for (int edge = ptr[row] + lane / width; edge < ptr[row + 1]; edge += 32 / width)
        if (col < columns) sum += val[edge] * x[idx[edge] * columns + col];
    for (uint offset = 16; offset >= width; offset /= 2)
        sum += simd_shuffle_down(sum, offset);
    if (lane < width && col < columns) out[row * columns + col] = sum;
}
''')


def graph_arrays(w):
    if max(w.shape[0], w.values().numel()) >= 2**31:
        raise ValueError("Metal graph indices exceed int32 capacity")
    return (w.crow_indices().to(device="mps", dtype=torch.int32),
            w.col_indices().to(device="mps", dtype=torch.int32),
            w.values().to(device="mps", dtype=torch.float32))


def multiply(x, ptr, idx, values):
    if x.dtype != torch.float32 or x.device.type != "mps" or x.ndim != 2:
        raise ValueError("Metal CSR requires a float32 matrix on MPS")
    x = x.contiguous()
    out = torch.empty((len(ptr)-1, x.shape[1]), dtype=x.dtype, device=x.device)
    width = min(32, 2**(x.shape[1]-1).bit_length())
    kernels().csr_mm(ptr, idx, values, x, out, x.shape[1], width,
                     threads=((len(ptr)-1)*32, (x.shape[1]+width-1)//width), group_size=(32, 1))
    return out


class MetalSparseMultiply(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, graph, transpose):
        ctx.transpose = transpose
        return multiply(x, *graph)

    @staticmethod
    def backward(ctx, grad):
        return multiply(grad, *ctx.transpose), None, None

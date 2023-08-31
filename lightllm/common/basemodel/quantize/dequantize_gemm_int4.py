import torch

import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 256}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 256}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 32}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 16}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 16}, num_stages=5, num_warps=8),
    ],
    key=['K', 'N'],
)


@triton.jit
def dequantize_kernel(
    # Pointers to matrices
    b_ptr, b_scale_ptr, b_zp_ptr, fpb_ptr,
    # Matrix dimensions
    K, N, group_size,
    stride_bk, stride_bn,
    stride_bsk, stride_bsn,
    stride_bzpk, stride_bzpn,
    stride_fpbk, stride_fpbn,
    # Meta-parameters
    BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    """Dequantize weight[K // 8, N], scale[K, N // 128], zp[K // 8, N // 128]
    """
    k_block_idx = tl.program_id(axis=0)
    n_block_idx = tl.program_id(axis=1)
    offs_k = k_block_idx * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    offs_n = n_block_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    b_offs = offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    bzp_offs = offs_k[:, None] * stride_bzpk + (offs_n // group_size)[None, :] * stride_bzpn
    n_mask = offs_n[None, :] < N
    k_mask = offs_k[:, None] < K
    mask = n_mask & k_mask
    int32_b = tl.load(b_ptr + b_offs, mask=mask, other=0.0)
    zp_b = tl.load(b_zp_ptr + bzp_offs, mask=mask, other=0.0)
    # Work on 8 rows once, this should be easily unrolled.
    for i in range(8):
        int4_b = ((int32_b << (28 - i * 4) >> 28) + 16) % 16
        int4_zp = ((zp_b << (28 - i * 4) >> 28) + 16) % 16
        bs_offs = (offs_k * 8 + i)[:, None] * stride_bsk + (offs_n // group_size)[None, :] * stride_bsn
        fpb_offs = (offs_k * 8 + i)[:, None] * stride_fpbk + offs_n[None, :] * stride_fpbn
        scale_b = tl.load(b_scale_ptr + bs_offs, mask=n_mask & ((offs_k * 8 + i)[:, None] < K * 8), other=0.0)
        fp_weight = (int4_b - int4_zp) * scale_b
        tl.store(fpb_ptr + fpb_offs, fp_weight, mask=n_mask & ((offs_k * 8 + i)[:, None] < K * 8))


def matmul_dequantize_int4(a, b, b_scale, b_zero_point, group_size, out=None):
    # Check constraints.
    assert a.is_contiguous(), "Matrix A must be contiguous"
    assert b.is_contiguous(), "Matrix B must be contiguous"
    M, K = a.shape
    Kw, N = b.shape
    if out == None:
        # Allocates output.
        c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    else:
        c = out
    fp_b = torch.empty((K, N), device=a.device, dtype=a.dtype)
    grid = lambda META: (
        triton.cdiv(Kw, META['BLOCK_SIZE_N']), triton.cdiv(N, META['BLOCK_SIZE_N']),
    )
    dequantize_kernel[grid](
        b, b_scale, b_zero_point, fp_b,
        Kw, N, group_size,
        b.stride(0), b.stride(1),
        b_scale.stride(0), b_scale.stride(1),
        b_zero_point.stride(0), b_zero_point.stride(1),
        fp_b.stride(0), fp_b.stride(1)
    )
    return torch.matmul(a, fp_b, out=c)


def quantize_int4(weight, group_size=128):
    # Weight shape: [H1, H2]
    # Scale shape: [H2]
    h1, h2 = weight.shape
    assert h1 % 8 == 0, "H1 {} H2 {}".format(h1, h2)
    assert h2 % group_size == 0, "H1 {} H2 {}".format(h1, h2)

    ori_weight = weight.clone()
    weight = weight.view(-1, group_size)
    weight_max = weight.amax(-1, keepdim=True)
    weight_max = torch.where(weight_max < 0, 0, weight_max)
    weight_min = weight.amin(-1, keepdim=True)
    weight_min = torch.where(weight_min > 0, 0, weight_min)
    weight_range = weight_max - weight_min 
    scale = weight_range / (2 ** 4 - 1)
    zero_point = (-weight_min / scale).round().clamp(0, 15).to(torch.int32)
    weight = (weight / scale + zero_point).clamp(0, 15).to(torch.int32).view(h1, h2)
    int_weight = torch.empty(h1 // 8, h2).to(torch.int32).to(weight.device)
    int_zero_point = torch.zeros(h1 // 8, h2 // group_size).to(torch.int32).to(weight.device)
    zero_point = zero_point.view(h1, -1)
    scale = scale.view(h1, -1)
    # pack 8 int4 in an int32 number.
    for pack in range(0, h1, 8):
        for i in range(8):
            int_weight[pack // 8, :] += weight[pack + i, :] << (i * 4)
            int_zero_point[pack // 8, :] += zero_point[pack + i, :] << (i * 4)
    '''
    fp_weight = torch.zeros(h1, h2).half().to(weight.device)
    for pack in range(0, h1 // 8):
        for i in range(8):
            fp_weight[pack * 8 + i, :] = \
                ((int_weight[pack, :] << (28 - i * 4) >> 28) + 16) % 16
    print((fp_weight - weight).abs().sum())

    fp_zp = torch.zeros(zero_point.shape).half().to(zero_point.device)
    for pack in range(0, h1 // 8):
        for i in range(8):
            fp_zp[pack * 8 + i, :] = \
                ((int_zero_point[pack, :] << (28 - i * 4) >> 28) + 16) % 16

    print((fp_zp - zero_point).abs().sum())
    '''
     
    return int_weight, scale, int_zero_point


def unpack_int4(weight, scale, zp, group_size, b):
    cos = torch.nn.CosineSimilarity(0)
    h1, h2 = weight.shape
    fp_weight = torch.zeros(h1 * 8, h2).half().to(weight.device)
    for pack in range(0, h1):
        for i in range(8):
            for j in range(h2 // group_size):
                unpack_weight = ((weight[pack, j * group_size:(j + 1) * group_size] << (28 - i * 4) >> 28) + 16) % 16
                unpack_zp = ((zp[pack, j] << (28 - i * 4) >> 28) + 16) % 16
                unpack_scale = scale[pack * 8 + i, j]
                fp_weight[pack * 8 + i, j * group_size:(j + 1) * group_size] = \
                    (unpack_weight - unpack_zp) * unpack_scale
    return fp_weight


def test_int4(M, K, N):
    import time

    print("M: {} K: {} N: {}".format(M, K, N))
    # test_correct_int4(M, K, N)
    # torch.manual_seed(0)
    a = torch.randn((M, K), device='cuda', dtype=torch.float16)
    b = torch.randn((K, N), device='cuda', dtype=torch.float16)
    int_b, b_scale, b_zero_point = quantize_int4(b)
    for _ in range(10):
        triton_output = matmul_dequantize_int4(a, int_b, b_scale, b_zero_point, 128)
    torch.cuda.synchronize()
    iters = 512
    t1 = time.time()
    for _ in range(iters):
        triton_output = matmul_dequantize_int4(a, int_b, b_scale, b_zero_point, 128)
    torch.cuda.synchronize()
    t2 = time.time()
    triton_time = t2 - t1
    print("Triton time cost", (t2 - t1))
    for _ in range(10):
        torch_output = torch.matmul(a, b)
    torch.cuda.synchronize()
    iters = 512
    t1 = time.time()
    for _ in range(iters):
        torch_output = torch.matmul(a, b)
    torch.cuda.synchronize()
    t2 = time.time()
    torch_time = t2 - t1
    print("Torch time cost", (t2 - t1))
    return triton_time, torch_time, 0


def test_correct_int4(M=512, K=4096, N=4096):
    import time

    a = torch.randn((M, K), device='cuda', dtype=torch.float16)
    b = torch.randn((K, N), device='cuda', dtype=torch.float16)
    int_b, b_scale, b_zero_point = quantize_int4(b)
    cos = torch.nn.CosineSimilarity(0)
    fp_weight = unpack_int4(int_b, b_scale, b_zero_point, 128, b)
    print("Quantize cos", cos(fp_weight.flatten().to(torch.float32), b.flatten().to(torch.float32)))
    triton_output = matmul_dequantize_int4(a, int_b, b_scale, b_zero_point, 128)
    torch_output = torch.matmul(a, b)
    print(f"triton_output={triton_output}")        
    print(f"torch_output={torch_output}")
    print("Output cos", cos(triton_output.flatten().to(torch.float32), torch_output.flatten().to(torch.float32)))


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['M', 'N', 'K'],  # Argument names to use as an x-axis for the plot
        x_vals=[
            128 * i for i in range(2, 33)
        ],  # Different possible values for `x_name`
        line_arg='provider',  # Argument name whose value corresponds to a different line in the plot
        # Possible values for `line_arg`
        line_vals=['cublas', 'triton'],
        # Label name for the lines
        line_names=["cuBLAS", "Triton"],
        # Line styles
        styles=[('green', '-'), ('blue', '-')],
        ylabel="TFLOPS",  # Label name for the y-axis
        plot_name="matmul-performance",  # Name for the plot, used also as a file name for saving the plot.
        args={},
    )
)
def benchmark(M, N, K, provider):
    a = torch.randn((M, K), device='cuda', dtype=torch.float16)
    b = torch.randn((K, N), device='cuda', dtype=torch.float16)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'cublas':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.matmul(a, b), quantiles=quantiles)
    if provider == 'triton':
        intb, b_scale, bzp = quantize_int4(b)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul_dequantize_int4(a, intb, b_scale, bzp, 128), quantiles=quantiles)
    perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
    return perf(ms), perf(max_ms), perf(min_ms)


def test_model_layer_merge(bs, sqe_len, hidden, inter, tp):
    st1 = 0
    st2 = 0
    t1, t2 = test_int4(bs * sqe_len, hidden, hidden * 3 // tp)
    st1 += t1
    st2 += t2
    t1, t2 = test_int4(bs * sqe_len, hidden // tp, hidden)
    st1 += t1
    st2 += t2
    t1, t2 = test_int4(bs * sqe_len, hidden, inter * 2 // tp)
    st1 += t1
    st2 += t2
    t1, t2 = test_int4(bs * sqe_len, inter // tp, hidden)
    st1 += t1
    st2 += t2
    print("Triton time {} Torch time {}".format(st1, st2))


if __name__ == "__main__":
    test_correct_int4()
    benchmark.run(show_plots=True, print_data=True)

    bs = 32
    hidden = 4096
    inter  = 11008
    prefill_len = 512
    decode_len = 1
    tp = 1
    test_model_layer(bs, prefill_len, hidden, inter, tp)
    test_model_layer(bs, decode_len, hidden, inter, tp)
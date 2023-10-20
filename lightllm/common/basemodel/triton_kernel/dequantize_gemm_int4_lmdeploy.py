import torch
from lightllm_lmdeploy_kernel import int4fp16_matmul, convert_s4_k_m8


def quantize_int4_lmdeploy(weight, group_size=128, pack_order=[0, 2, 4, 6, 1, 3, 5, 7]):
    """
    weight: [K, N]
    return:
        qweight: [K, N//8] int32 (packed int4*8) new pack_order
        scale_zeros: [K//group_size, N] int32
        # qzeros: [K//group_size, N//8] int32 (packed int4*8) new pack_order
    """
    K, N = weight.shape
    weight = weight.transpose(1, 0)
    print("quantize_int4_lmdeploy for K={} N={} ...".format(K, N))
    assert K % 8 == 0 and N % 8 == 0, "K={} N={}".format(K, N)
    assert K % group_size == 0, "K={} N={}".format(K, N)

    weight = weight.contiguous().view(-1, group_size).cuda()
    weight_max = weight.amax(-1, keepdim=True)
    weight_max = torch.where(weight_max < 0, 0, weight_max)
    weight_min = weight.amin(-1, keepdim=True)
    weight_min = torch.where(weight_min > 0, 0, weight_min)
    weight_range = weight_max - weight_min 
    
    scale = (weight_range / (2 ** 4 - 1))
    zero_point = (-weight_min / scale).round().clamp(0, 15).to(torch.int32)
    # (N, K)
    weight = (weight / scale + zero_point).round().clamp(0, 15).to(torch.int32).view(N, K)
    # (N, K//group_size)
    scale = scale.view(N, -1)
    # (N, K//group_size)
    zero_point = zero_point.view(N, -1)

    # pack 8 int4 in an int32 number at axis-N
    qweight = torch.zeros((N // 8, K), dtype=torch.int32, device=weight.device)
    qzeros  = torch.zeros((N // 8, K // group_size), dtype=torch.int32, device=weight.device)

    for pack in range(0, N, 8):
        for i in range(8):
            qweight[pack // 8, :] += weight[pack + pack_order[i], :] << (i * 4)
            qzeros[pack // 8, :] += zero_point[pack + pack_order[i], :] << (i * 4)

    weight = None
    qweight = qweight.transpose(1, 0).contiguous()
    scale = scale.transpose(1, 0).contiguous()
    qzeros = qzeros.transpose(1, 0).contiguous()

    # convert to layout defined inside lmdeploy
    qweight_new = torch.zeros_like(qweight)
    scale_zeros = torch.zeros_like(scale, dtype=torch.int32)  # half2
    temp = torch.zeros_like(scale)
    convert_s4_k_m8(
        qweight_new,
        scale_zeros,
        temp,
        qweight,
        scale,
        qzeros,
        N,
        K,
        group_size
    )
    temp = None
    scale = None
    return qweight_new, None, scale_zeros


def matmul_dequantize_int4_lmdeploy(
        x: torch.FloatTensor,
        qweight: torch.IntTensor,
        scales: torch.FloatTensor,
        scale_zeros: torch.IntTensor,
        group_size,
        output
) -> torch.FloatTensor:
    """
    x is activation:             (M, K) float16
    qweight is quant weight:     (K, N//8) int32 (int4*8 packed with pack_order)
    scales: not used, should be None, only for unified input format
    scale_zeros is quant merged(scales, qzeros):      (K//group_size, N) int32
    return tensor:               (M, N) float16
    """
    assert x.shape[1] == qweight.shape[0], "A must be a multiple of 8 in the last dimension"
    M, K = x.shape
    N = qweight.shape[1] * 8
    int4fp16_matmul(output, qweight, x, scale_zeros, N, M, K, group_size, False)
    return output


def test_correct(M=32, K=2048, N=4096):
    from lightllm.common.basemodel.triton_kernel.dequantize_gemm_int4 import quantize_int4, unpack_int4, matmul_dequantize_int4_gptq

    cos = torch.nn.CosineSimilarity(0)
    def cmp(x, y):
        return cos(x.flatten().to(torch.float32), y.flatten().to(torch.float32))
    def mp(x, s):
        print("{}: {} {} {}".format(s, x.dtype, x.shape, x))

    group_size = 128
    a = torch.randn((M, K), device='cuda', dtype=torch.float16) # (M, K)
    w = torch.randn((K, N), device='cuda', dtype=torch.float16) # (K, N)

    torch_output = torch.matmul(a, w)
    mp(torch_output, "torch_output_fp16")

    qweight1, scales1, qzeros1 = quantize_int4_lmdeploy(w, group_size=group_size) # (K, N//8), (K//group_size, N), (K//group_size, N//8)
    lm_output = torch.empty((M, N), device=a.device, dtype=torch.float16)
    lm_output = matmul_dequantize_int4_lmdeploy(a, qweight1, scales1, qzeros1, group_size, lm_output)
    mp(lm_output, "lm_output_int4")
    print("Output cos(lmdeploy_int4, torch_fp16):", cmp(lm_output, torch_output))

    qweight2, scales2, qzeros2 = quantize_int4(w, group_size=group_size) # (K//8, N), (K//group_size, N), (K//group_size, N//8)
    triton_output = matmul_dequantize_int4_gptq(a, qweight2, scales2, qzeros2, group_size)
    mp(triton_output, "triton_output_int4")
    print("Output cos(triton_int4, torch_fp16):", cmp(triton_output, torch_output))
    print("Output cos(triton_int4, lmdeploy_int4):", cmp(triton_output, lm_output))


if __name__ == "__main__":
    test_correct(M=32, K=2048, N=4096)
    test_correct(M=1, K=2048, N=4096)

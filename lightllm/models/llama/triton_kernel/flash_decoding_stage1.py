import torch

import triton
import triton.language as tl
import math


configs = []
for n in [2 ** i  for i in range(9)]:
    for seq_block in [2 ** i for i in range(5, 9)]:
        for num_stages in [ 1, 2, 3, 4]:
            for num_wraps in [1, 2, 4, 8]:
                if seq_block > n and seq_block % n == 0:
                    configs.append(
                        triton.Config(
                            { 
                                'BLOCK_N': n,
                                'BLOCK_SEQ': seq_block,
                                'BLOCK_DMODEL' : 128
                            },
                            num_stages=num_stages, num_warps=num_wraps
                        )
                    )


@triton.jit
def _fwd_kernel_flash_decode_stage1(
    Q, K, V, sm_scale, B_Loc, B_Seqlen, batch_size, max_input_len,
    Mid_O, # [batch, head, seq_block_num, head_dim]
    Mid_O_LogExpSum, #[batch, head, seq_block_num]
    stride_b_loc_b, stride_b_loc_s,
    stride_qbs, stride_qh, stride_qd,
    stride_kbs, stride_kh, stride_kd,
    stride_vbs, stride_vh, stride_vd,
    stride_mid_ob, stride_mid_oh, stride_mid_os, stride_mid_od,
    stride_mid_o_eb, stride_mid_o_eh, stride_mid_o_es,
    BLOCK_SEQ: tl.constexpr, 
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    seq_start_block = tl.program_id(2)

    offs_d = tl.arange(0, BLOCK_DMODEL)
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_start_index = max_input_len - cur_batch_seq_len + seq_start_block * BLOCK_SEQ
    cur_batch_end_index = tl.minimum(max_input_len, cur_batch_start_index + BLOCK_SEQ)

    off_q = cur_batch * stride_qbs + cur_head * stride_qh + offs_d
    
    block_n_size = tl.where(cur_batch_end_index - cur_batch_start_index <= 0, 0, cur_batch_end_index - cur_batch_start_index + BLOCK_N - 1) // BLOCK_N
    
    offs_n = cur_batch_start_index + tl.arange(0, BLOCK_N)
    
    q = tl.load(Q + off_q)

    sum_exp = 0.0
    max_logic = -float("inf")
    acc = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)

    for start_n in range(0, block_n_size, 1):
        offs_n_new = start_n * BLOCK_N + offs_n
        k_loc = tl.load(B_Loc + stride_b_loc_b * cur_batch +  offs_n_new, mask=offs_n_new < cur_batch_end_index, other=0)
        off_k = k_loc[:, None] * stride_kbs + cur_head * stride_kh + offs_d[None, :]
        k = tl.load(K + off_k, mask=offs_n_new[:, None] < cur_batch_end_index, other=0.0)
        att_value = tl.sum(q[None, :] * k, 1)
        att_value *= sm_scale
        att_value = tl.where(offs_n_new < cur_batch_end_index, att_value, float("-inf"))
        v = tl.load(V + off_k, mask=offs_n_new[:, None] < cur_batch_end_index, other=0.0)
        
        cur_max_logic = tl.max(att_value, axis=0)
        new_max_logic = tl.maximum(cur_max_logic, max_logic)

        exp_logic = tl.exp(att_value - new_max_logic)
        logic_scale = tl.exp(max_logic - new_max_logic)
        acc *= logic_scale
        acc += tl.sum(exp_logic[:, None] * v, axis=0)

        sum_exp = sum_exp * logic_scale + tl.sum(exp_logic, axis=0)
        max_logic = new_max_logic
    
    need_store = tl.where(block_n_size == 0, 0, 1)
    for _ in range(0, need_store, 1):
        off_mid_o = cur_batch * stride_mid_ob + cur_head * stride_mid_oh + seq_start_block * stride_mid_os + offs_d
        off_mid_o_logexpsum = cur_batch * stride_mid_o_eb + cur_head * stride_mid_o_eh + seq_start_block
        tl.store(Mid_O + off_mid_o, acc / sum_exp)
        tl.store(Mid_O_LogExpSum + off_mid_o_logexpsum, max_logic + tl.log(sum_exp))
    return


# code based https://github.com/fpgaminer/GPTQ-triton
auto_fwd_kernel_flash_decode_stage1 = triton.autotune(
    configs=configs,
    key=['batch_size', 'max_input_len'],
    warmup=1, rep=20
    # nearest_power_of_two=True,
    # prune_configs_by={
    #     'early_config_prune': custom_autotune.matmul248_kernel_config_pruner,
    #     'perf_model': None,
    #     'top_k': None,
    # },
)(_fwd_kernel_flash_decode_stage1)

@torch.no_grad()
def flash_decode_stage1(q, k, v, B_Loc, B_Seqlen, max_len_in_batch, mid_out, mid_out_logsumexp, BLOCK_SEQ = 128, BLOCK_N = 32):
    assert BLOCK_SEQ % BLOCK_N == 0
    # shape constraints
    Lq, Lk = q.shape[-1], k.shape[-1]
    assert Lq == Lk
    assert Lk in {16, 32, 64, 128}
    sm_scale = 1.0 / (Lk ** 0.5)
    batch, head_num = B_Loc.shape[0], q.shape[1]
    # grid = (batch, head_num, triton.cdiv(max_len_in_batch, BLOCK_SEQ))

    grid = lambda META: (batch, head_num, triton.cdiv(max_len_in_batch, META['BLOCK_SEQ']))
    # num_warps = 4
    
    auto_fwd_kernel_flash_decode_stage1[grid](
        q, k, v, sm_scale, B_Loc, B_Seqlen, batch, max_len_in_batch,
        mid_out,
        mid_out_logsumexp,
        B_Loc.stride(0), B_Loc.stride(1),
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        mid_out.stride(0), mid_out.stride(1), mid_out.stride(2), mid_out.stride(3),
        mid_out_logsumexp.stride(0), mid_out_logsumexp.stride(1), mid_out_logsumexp.stride(2),
        # BLOCK_SEQ=BLOCK_SEQ,
        # BLOCK_DMODEL=Lk,
        # BLOCK_N=BLOCK_N,
        # num_warps=num_warps,
        # num_stages=1,
    )
    return

"""
for orin 1, 128, 32, 128
+++ triton autotuner cache key: BLOCK_N: 16, BLOCK_SEQ: 32, BLOCK_DMODEL: 128, num_warps: 1, num_ctas: 1, num_stages: 2, enable_warp_specialization: False, enable_persistent: False
"""
@torch.no_grad()
def flash_decode_stage1_outer(q, k, v, B_Loc, B_Seqlen, max_len_in_batch, mid_out, mid_out_logsumexp):
    BLOCK_SEQ = 32 # 256
    BLOCK_N = 16
    assert BLOCK_SEQ % BLOCK_N == 0
    # shape constraints
    Lq, Lk = q.shape[-1], k.shape[-1]
    assert Lq == Lk
    assert Lk in {16, 32, 64, 128}
    sm_scale = 1.0 / (Lk ** 0.5)
    batch, head_num = B_Loc.shape[0], q.shape[1]
    grid = (batch, head_num, triton.cdiv(max_len_in_batch, BLOCK_SEQ))
    BLs0, BLs1 = B_Loc.stride()
    qs0, qs1, qs2 = q.stride()
    ks0, ks1, ks2 = k.stride()
    vs0, vs1, vs2 = v.stride()
    mo_s0, mo_s1, mo_s2, mo_s3 = mid_out.stride()
    mol_s0, mol_s1, mol_s2 = mid_out_logsumexp.stride()
    
    _fwd_kernel_flash_decode_stage1[grid](
        q, k, v, sm_scale, B_Loc, B_Seqlen, batch, max_len_in_batch,
        mid_out,
        mid_out_logsumexp,
        BLs0, BLs1,
        qs0, qs1, qs2,
        ks0, ks1, ks2,
        vs0, vs1, vs2,
        mo_s0, mo_s1, mo_s2, mo_s3,
        mol_s0, mol_s1, mol_s2,
        BLOCK_SEQ=BLOCK_SEQ,
        BLOCK_DMODEL=128,
        BLOCK_N=BLOCK_N,
        num_warps=1,
        num_stages=2, #2,
    )
    return


def test1():
    import time

    B, N_CTX, H, D = 1, 128, 32, 128

    dtype = torch.float16

    q = torch.empty((B, H, D), dtype=dtype, device="cuda").normal_(mean=0.1, std=0.2)
    k = torch.empty((B * N_CTX, H, D), dtype=dtype, device="cuda").normal_(mean=0.1, std=0.2)
    v = torch.empty((B * N_CTX, H, D), dtype=dtype, device="cuda").normal_(mean=0.1, std=0.2)

    mid_out = torch.empty((B, H, 2048 * 8, D), dtype=torch.float32, device="cuda")
    mid_out_logexpsum = torch.empty((B, H, 2048 * 8), dtype=torch.float32, device="cuda")

    b_loc = torch.zeros((B, N_CTX), dtype=torch.int32, device="cuda")
    b_start_loc = torch.zeros((B,), dtype=torch.int32, device="cuda")
    b_seq_len = torch.zeros((B,), dtype=torch.int32, device="cuda")

    for i in range(B):
        b_start_loc[i] = i * N_CTX
        b_seq_len[i] = N_CTX
        b_loc[i] = i * N_CTX + torch.arange(0, N_CTX, dtype=torch.int32, device="cuda")
        # print(b_loc[i])

    # Warm up
    for _ in range(10):
        flash_decode_stage1(q, k, v, b_loc, b_seq_len, N_CTX, mid_out, mid_out_logexpsum)
    run_iter = 1000
    torch.cuda.synchronize()
    t1 = time.time()
    for _ in range(run_iter):
         flash_decode_stage1(q, k, v, b_loc, b_seq_len, N_CTX, mid_out, mid_out_logexpsum)
    torch.cuda.synchronize()
    t2 = time.time()
    print("Time cost {}".format((t2 - t1) / run_iter))

    # torch_out = torch_att(q, k, B, N_CTX, H, D).squeeze()
    # o = att_out.squeeze()
    # print("max ", torch.max(torch.abs(torch_out - o)))
    # print("mean ", torch.mean(torch.abs(torch_out - o)))
    # assert torch.allclose(torch_out, o, atol=1e-2, rtol=0)


if __name__ == '__main__':
    test1()

import torch

import triton
import triton.language as tl
import math



@triton.jit
def _fwd_kernel_flash_decode_stage2(
    B_Seqlen,
    Mid_O, # [batch, head, seq_block_num, head_dim]
    Mid_O_LogExpSum, #[batch, head, seq_block_num]
    O, #[batch, head, head_dim]
    stride_mid_ob, stride_mid_oh, stride_mid_os, stride_mid_od,
    stride_mid_o_eb, stride_mid_o_eh, stride_mid_o_es,
    stride_obs, stride_oh, stride_od,
    BLOCK_SEQ: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)

    offs_d = tl.arange(0, BLOCK_DMODEL)
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)

    block_n_size = tl.where(cur_batch_seq_len <= 0, 0, cur_batch_seq_len + BLOCK_SEQ - 1) // BLOCK_SEQ

    sum_exp = 0.0
    max_logic = -float("inf")
    acc = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)

    offs_v = cur_batch * stride_mid_ob + cur_head * stride_mid_oh + offs_d
    offs_logic = cur_batch * stride_mid_o_eb + cur_head * stride_mid_o_eh
    for block_seq_n in range(0, block_n_size, 1):
        tv = tl.load(Mid_O + offs_v + block_seq_n * stride_mid_os)
        tlogic = tl.load(Mid_O_LogExpSum + offs_logic + block_seq_n)
        new_max_logic = tl.maximum(tlogic, max_logic)
        
        old_scale = tl.exp(max_logic - new_max_logic)
        acc *= old_scale
        exp_logic = tl.exp(tlogic - new_max_logic)
        acc += exp_logic * tv
        sum_exp = sum_exp * old_scale + exp_logic
        max_logic = new_max_logic
    
    tl.store(O + cur_batch * stride_obs + cur_head * stride_oh + offs_d, acc / sum_exp)
    return


configs = []
# for n in [2 ** i  for i in range(11)]:
# for seq_block in [2 ** i for i in range(11)]:
for seq_block in [32]:
    for num_stages in [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]:
        for num_wraps in [1, 2, 4, 8]:
            # if seq_block > n and seq_block % n == 0:
            configs.append(
                triton.Config(
                    { 
                        # 'BLOCK_N': n,
                        'BLOCK_SEQ': seq_block,
                        'BLOCK_DMODEL' : 128
                    },
                    num_stages=num_stages, num_warps=num_wraps
                )
            )

# code based https://github.com/fpgaminer/GPTQ-triton
auto_fwd_kernel_flash_decode_stage2 = triton.autotune(
    configs=configs,
    key=[],
    warmup=1, rep=10
    # nearest_power_of_two=True,
    # prune_configs_by={
    #     'early_config_prune': custom_autotune.matmul248_kernel_config_pruner,
    #     'perf_model': None,
    #     'top_k': None,
    # },
)(_fwd_kernel_flash_decode_stage2)

@torch.no_grad()
def flash_decode_stage2(mid_out, mid_out_logexpsum, B_Seqlen, O, BLOCK_SEQ = 128):
    Lk = mid_out.shape[-1]
    assert Lk in {16, 32, 64, 128}
    batch, head_num = mid_out.shape[0], mid_out.shape[1]
    # grid = (batch, head_num, triton.cdiv(max_len_in_batch, BLOCK_SEQ))
    grid = lambda META: (batch, head_num)
    
    auto_fwd_kernel_flash_decode_stage2[grid](
        B_Seqlen, mid_out, mid_out_logexpsum, O,
        mid_out.stride(0), mid_out.stride(1), mid_out.stride(2), mid_out.stride(3),
        mid_out_logexpsum.stride(0), mid_out_logexpsum.stride(1), mid_out_logexpsum.stride(2),
        O.stride(0), O.stride(1), O.stride(2)
        # BLOCK_SEQ=BLOCK_SEQ,
        # BLOCK_DMODEL=Lk,
        # BLOCK_N=BLOCK_N,
        # num_warps=num_warps,
        # num_stages=1,
    )
    return

"""
for orin 1, 128, 32, 128
+++ triton autotuner cache key: BLOCK_SEQ: 32, BLOCK_DMODEL: 128, num_warps: 2, num_ctas: 1, num_stages: 9, enable_warp_specialization: False, enable_persistent: False
"""
@torch.no_grad()
def flash_decode_stage2_outer(mid_out, mid_out_logexpsum, B_Seqlen, O):
    Lk = mid_out.shape[-1]
    assert Lk in {16, 32, 64, 128}
    batch, head_num = mid_out.shape[0], mid_out.shape[1]
    grid = (batch, head_num)
    # grid = lambda META: (batch, head_num)
    mo_s0, mo_s1, mo_s2, mo_s3 = mid_out.stride()
    mol_s0, mol_s1, mol_s2 = mid_out_logexpsum.stride()
    Os0, Os1, Os2 = O.stride()
    
    _fwd_kernel_flash_decode_stage2[grid](
        B_Seqlen, mid_out, mid_out_logexpsum, O,
        mo_s0, mo_s1, mo_s2, mo_s3,
        mol_s0, mol_s1, mol_s2,
        Os0, Os1, Os2,
        BLOCK_SEQ= 32, #256,
        BLOCK_DMODEL=128,
        num_warps= 2, #1,
        num_stages=9, #3,
    )
    return


def test1():
    import time

    B, N_CTX, H, D = 1, 128, 32, 128

    dtype = torch.float16

    o = torch.empty((B, H, D), dtype=dtype, device="cuda").normal_(mean=0.1, std=0.2)

    mid_out = torch.empty((B, H, 2048 * 8, D), dtype=torch.float32, device="cuda")
    mid_out_logexpsum = torch.empty((B, H, 2048 * 8), dtype=torch.float32, device="cuda")

    b_seq_len = torch.zeros((B,), dtype=torch.int32, device="cuda")

    for i in range(B):
        b_seq_len[i] = N_CTX

    # Warm up
    for _ in range(10):
        flash_decode_stage2(mid_out, mid_out_logexpsum, b_seq_len, o)
    run_iter = 1000
    torch.cuda.synchronize()
    t1 = time.time()
    for _ in range(run_iter):
         flash_decode_stage2(mid_out, mid_out_logexpsum, b_seq_len, o)
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

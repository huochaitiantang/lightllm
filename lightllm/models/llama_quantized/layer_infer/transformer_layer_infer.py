from typing import Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.functional as F
import triton
import os

from lightllm.models.llama.layer_weights.transformer_layer_weight import LlamaTransformerLayerWeight
from lightllm.models.llama_quantized.layer_weights.transformer_layer_weight import LlamaTransformerLayerWeightQuantized
from lightllm.models.llama.triton_kernel.context_flashattention_nopad import context_attention_fwd
from lightllm.models.llama.triton_kernel.token_attention_nopad_att1 import token_att_fwd, token_att_fwd_int8k
from lightllm.models.llama.triton_kernel.token_attention_nopad_softmax import token_softmax_fwd
from lightllm.models.llama.triton_kernel.token_attention_nopad_reduceV import token_att_fwd2, token_att_fwd2_int8v
from lightllm.models.llama.triton_kernel.rmsnorm import rmsnorm_forward
from lightllm.models.llama.triton_kernel.rotary_emb import rotary_emb_fwd
from lightllm.models.llama.infer_struct import LlamaInferStateInfo
from lightllm.models.llama.layer_infer.transformer_layer_infer import LlamaTransformerLayerInfer
from lightllm.common.basemodel.triton_kernel.destindex_copy_kv import destindex_copy_kv, destindex_copy_quantize_kv
from lightllm.common.basemodel.triton_kernel.quantize_gemm_int8 import matmul_quantize_int8, silu_mul
from lightllm.common.basemodel.triton_kernel.dequantize_gemm_int8 import matmul_dequantize_int8
from lightllm.common.basemodel.triton_kernel.dequantize_gemm_int4 import matmul_dequantize_int4_s1, matmul_dequantize_int4_s2, matmul_dequantize_int4_gptq
from lightllm.utils.infer_utils import mark_cost_time
from lightllm.models.llama.triton_kernel.token_attention_softmax_and_reducev import token_softmax_reducev_fwd
from lightllm.common.basemodel.triton_kernel.dequantize_gemm_int4_lmdeploy import matmul_dequantize_int4_lmdeploy

DEVICE = torch.cuda.current_device()
MAX_BATCH = int(os.getenv('MAX_BATCH', 5))
MAX_TOKEN = int(os.getenv('MAX_BATCH_TOTAL_TOKENS', 4096))

# common temp tensors
T = {}
ATTN_M = torch.empty((32, MAX_TOKEN), device=DEVICE, dtype=torch.float16)
ATTN_O = torch.empty((MAX_BATCH, 32, 128), device=DEVICE, dtype=torch.float16)
NORM_O = torch.empty((MAX_BATCH, 4096), device=DEVICE, dtype=torch.float16)
MATMUL_A = torch.empty((MAX_BATCH, 11008), device=DEVICE, dtype=torch.int8)
MATMUL_AS = torch.empty(MAX_BATCH, device=DEVICE, dtype=torch.float16)
MATMUL_O = torch.empty((MAX_BATCH, 22016), device=DEVICE, dtype=torch.float16)
SILUMUL_O = torch.empty((MAX_BATCH, 11008), device=DEVICE, dtype=torch.float16)


def getT(dims, dtype, init_tensor):
    global T
    key = (dims, dtype)
    if key in T:
        return T[key]
    dim_cnt = len(dims)
    if dim_cnt == 2:
        T[key] = init_tensor[:dims[0], :dims[1]].view(*dims)
    elif dim_cnt == 3:
        T[key] = init_tensor[:dims[0], :dims[1], :dims[2]].view(*dims)
    elif dim_cnt == 1:
        T[key] = init_tensor[:dims[0]].view(*dims)
    else:
        raise
    return T[key]

 
class LlamaTransformerLayerInferINT8(LlamaTransformerLayerInfer):
    """
    Llama Model Inference using Triton W8A8 or W8A16 kernel.
    When prefill, we use `matmul_quantize_int8`, and use `matmul_dequantize_int8` when decode.
    For better balance latency and accurcy.
    """

    def __init__(self, layer_num, tp_rank, world_size, network_config, mode=[], group_size=None):
        super().__init__(layer_num, tp_rank, world_size, network_config, mode)
        self.inter_dim_ = network_config['intermediate_size']
        assert self.tp_q_head_num_ == self.tp_k_head_num_
        assert self.tp_q_head_num_ == self.tp_v_head_num_

    def _get_qkv_context(self, input, infer_state: LlamaInferStateInfo, layer_weight: LlamaTransformerLayerWeightQuantized) -> torch.Tensor:
        matmul_int8_func = matmul_dequantize_int8
        M, K = input.shape
        assert K == self.embed_dim_
        qkv_output = matmul_int8_func(
            input,
            layer_weight.qkv_fused_weight,
            layer_weight.qkv_fused_weight_scale
        )
        tmp_qkv = qkv_output.view(-1, 3, self.tp_q_head_num_, self.head_dim_)
        q = tmp_qkv[:, 0, :, :]
        cache_k_ = tmp_qkv[:, 1, :, :]
        cache_v_ = tmp_qkv[:, 2, :, :]

        rotary_emb_fwd(q, infer_state.position_cos, infer_state.position_sin)
        rotary_emb_fwd(cache_k_, infer_state.position_cos, infer_state.position_sin)
        return q, cache_k_, cache_v_

    def _get_qkv_decode(self, input, infer_state: LlamaInferStateInfo, layer_weight: LlamaTransformerLayerWeightQuantized) -> torch.Tensor:
        matmul_int8_func = matmul_quantize_int8
        M, K = input.shape
        assert K == self.embed_dim_
        N = layer_weight.qkv_fused_weight.shape[1]
        qkv_output = matmul_int8_func(
            input,
            layer_weight.qkv_fused_weight,
            layer_weight.qkv_fused_weight_scale,
            getT((M, K), "int8", MATMUL_A),
            getT((M,), "fp16", MATMUL_AS),
            getT((M, N), "fp16", MATMUL_O),
        )
        tmp_qkv = qkv_output.view(-1, 3, self.tp_q_head_num_, self.head_dim_)
        q = tmp_qkv[:, 0, :, :]
        cache_k_ = tmp_qkv[:, 1, :, :]
        cache_v_ = tmp_qkv[:, 2, :, :]

        rotary_emb_fwd(q, infer_state.position_cos, infer_state.position_sin)
        rotary_emb_fwd(cache_k_, infer_state.position_cos, infer_state.position_sin)
        return q, cache_k_, cache_v_

    def _get_o_context(self, input, infer_state: LlamaInferStateInfo, layer_weight: LlamaTransformerLayerWeightQuantized) -> torch.Tensor:
        o_tensor = matmul_dequantize_int8(input.view(-1, self.tp_o_head_num_ * self.head_dim_),
                                          layer_weight.o_weight_, layer_weight.o_weight_scale_)
        return o_tensor

    def _get_o_decode(self, input, infer_state: LlamaInferStateInfo, layer_weight: LlamaTransformerLayerWeightQuantized) -> torch.Tensor:
        new_input = input.view(-1, self.tp_o_head_num_ * self.head_dim_)
        M, K = new_input.shape
        N = layer_weight.o_weight_.shape[1]
        o_tensor = matmul_quantize_int8(
            new_input,
            layer_weight.o_weight_,
            layer_weight.o_weight_scale_,
            getT((M, K), "int8", MATMUL_A),
            getT((M,), "fp16", MATMUL_AS),
            getT((M, N), "fp16", MATMUL_O),
        )
        return o_tensor

    def _att_norm(self, input, infer_state:LlamaInferStateInfo, layer_weight:LlamaTransformerLayerWeight, output=None)->torch.Tensor:
        return rmsnorm_forward(input, weight=layer_weight.att_norm_weight_, eps=self.eps_, y=output)

    def _ffn_norm(self, input, infer_state:LlamaInferStateInfo, layer_weight:LlamaTransformerLayerWeight, output=None)->torch.Tensor:
        return rmsnorm_forward(input, weight=layer_weight.ffn_norm_weight_, eps=self.eps_, y=output)

    def _ffn_context(self, input, infer_state: LlamaInferStateInfo, layer_weight: LlamaTransformerLayerWeightQuantized) -> torch.Tensor:
        matmul_int8_func = matmul_dequantize_int8
        M, K = input.shape
        assert K == self.embed_dim_
        gate_up_output = matmul_int8_func(
            input,
            layer_weight.gate_up_fused_weight,
            layer_weight.gate_up_fused_weight_scale
        )
        input = None
        tp_inter_dim = self.inter_dim_ // self.world_size_
        gate_up_output = gate_up_output.view(-1, 2, tp_inter_dim)
        ffn1_out = torch.empty_like(gate_up_output[:, 0])
        ffn1_out = silu_mul(gate_up_output[:, 0], gate_up_output[:, 1], ffn1_out)
        gate_up_output = None
        ffn2_out = matmul_int8_func(ffn1_out, layer_weight.down_proj, layer_weight.down_proj_scale)
        ffn1_out = None
        return ffn2_out

    def _ffn_decode(self, input, infer_state: LlamaInferStateInfo, layer_weight: LlamaTransformerLayerWeightQuantized) -> torch.Tensor:
        matmul_int8_func = matmul_quantize_int8
        M, K = input.shape
        assert K == self.embed_dim_
        N1 = layer_weight.gate_up_fused_weight.shape[1]
        gate_up_output = matmul_int8_func(
            input,
            layer_weight.gate_up_fused_weight,
            layer_weight.gate_up_fused_weight_scale,
            getT((M, K), "int8", MATMUL_A),
            getT((M,), "fp16", MATMUL_AS),
            getT((M, N1), "fp16", MATMUL_O),
        )
        input = None
        tp_inter_dim = self.inter_dim_ // self.world_size_
        gate_up_output = gate_up_output.view(-1, 2, tp_inter_dim)
        B = gate_up_output.shape[0]

        ffn1_out = silu_mul(
            gate_up_output[:, 0],
            gate_up_output[:, 1],
            getT((B, tp_inter_dim), "fp16", SILUMUL_O),
        )

        N2 = layer_weight.down_proj.shape[1]
        ffn2_out = matmul_int8_func(
            ffn1_out,
            layer_weight.down_proj,
            layer_weight.down_proj_scale,
            getT((B, tp_inter_dim), "int8", MATMUL_A),
            getT((B,), "fp16", MATMUL_AS),
            getT((B, N2), "fp16", MATMUL_O),
        )
        return ffn2_out

    @mark_cost_time("trans context flash forward time cost")  # dont to remove this, will make performence down, did not know why
    def _context_attention(self, input_embding, infer_state: LlamaInferStateInfo, layer_weight):
        input1 = self._att_norm(input_embding, infer_state, layer_weight)
        self._pre_cache_kv(infer_state, layer_weight)
        q, cache_k, cache_v = self._get_qkv_context(input1, infer_state, layer_weight)
        input1 = None
        self._post_cache_kv(cache_k, cache_v, infer_state, layer_weight)
        o = self._context_attention_kernel(q, cache_k, cache_v, infer_state, layer_weight)
        q = None
        o = self._get_o_context(o, infer_state, layer_weight)
        if self.world_size_ > 1:
            dist.all_reduce(o, op=dist.ReduceOp.SUM, async_op=False)
        input_embding.add_(o.view(-1, self.embed_dim_))

    @mark_cost_time("trans context ffn forward time cost")  # dont to remove this, will make performence down, did not know why
    def _context_ffn(self, input_embdings, infer_state: LlamaInferStateInfo, layer_weight):
        input1 = self._ffn_norm(input_embdings, infer_state, layer_weight)
        ffn_out = self._ffn_context(input1, infer_state, layer_weight)
        input1 = None
        if self.world_size_ > 1:
            dist.all_reduce(ffn_out, op=dist.ReduceOp.SUM, async_op=False)
        input_embdings.add_(ffn_out.view(-1, self.embed_dim_))

    def _token_attention(self, input_embding, infer_state: LlamaInferStateInfo, layer_weight):
        B, E = input_embding.shape
        norm_o = getT((B, E), "fp16", NORM_O)
        input1 = self._att_norm(input_embding, infer_state, layer_weight, norm_o)
        self._pre_cache_kv(infer_state, layer_weight)
        q, cache_k, cache_v = self._get_qkv_decode(input1, infer_state, layer_weight)
        input1 = None
        self._post_cache_kv(cache_k, cache_v, infer_state, layer_weight)
        o = self._token_attention_kernel(q, infer_state, layer_weight)
        q = None
        o = self._get_o_decode(o, infer_state, layer_weight)
        if self.world_size_ > 1:
            dist.all_reduce(o, op=dist.ReduceOp.SUM, async_op=False)
        input_embding.add_(o.view(-1, self.embed_dim_))

    def _token_ffn(self, input_embdings, infer_state: LlamaInferStateInfo, layer_weight):
        B, E = input_embdings.shape
        norm_o = getT((B, E), "fp16", NORM_O)
        input1 = self._ffn_norm(input_embdings, infer_state, layer_weight, norm_o)
        ffn_out = self._ffn_decode(input1, infer_state, layer_weight)
        input1 = None
        if self.world_size_ > 1:
            dist.all_reduce(ffn_out, op=dist.ReduceOp.SUM, async_op=False)
        input_embdings.add_(ffn_out.view(-1, self.embed_dim_))
    
    # Pre / post cache kv for fused weight.
    def _pre_cache_kv(self, infer_state: LlamaInferStateInfo, layer_weight)->Tuple[torch.Tensor, torch.Tensor]:
        '''
        Release kv buffer to save memory, since we allocate while kv projection.
        '''
        if infer_state.is_prefill:
            infer_state.prefill_key_buffer = None
            infer_state.prefill_value_buffer = None
        else:
            infer_state.decode_key_buffer = None
            infer_state.decode_value_buffer = None

    def _post_cache_kv(self, cache_k, cache_v, infer_state: LlamaInferStateInfo, layer_weight: LlamaTransformerLayerWeightQuantized):
        mem_manager = infer_state.mem_manager
        if infer_state.is_prefill:
            self._copy_kv_to_mem_cache(cache_k, cache_v, infer_state.prefill_mem_index, mem_manager)
        else:
            self._copy_kv_to_mem_cache(cache_k, cache_v, infer_state.decode_mem_index, mem_manager)

    def _token_decode_attention_normal(self, q, infer_state: LlamaInferStateInfo):
        total_token_num = infer_state.total_token_num
        batch_size = infer_state.batch_size
        att_m_tensor = getT((self.tp_q_head_num_, total_token_num), "fp16", ATTN_M)
        token_att_fwd(q,
                      infer_state.mem_manager.key_buffer[self.layer_num_],
                      att_m_tensor,
                      infer_state.b_loc,
                      infer_state.b_start_loc,
                      infer_state.b_seq_len,
                      infer_state.max_len_in_batch)

        if triton.__version__ >= "2.1.0":
            o_tensor = getT((batch_size, self.tp_q_head_num_, self.head_dim_), "fp16", ATTN_O)
            token_softmax_reducev_fwd(att_m_tensor,
                                      infer_state.mem_manager.value_buffer[self.layer_num_],
                                      o_tensor,
                                      infer_state.b_loc,
                                      infer_state.b_start_loc,
                                      infer_state.b_seq_len,
                                      infer_state.max_len_in_batch,
                                      infer_state.other_kv_index)
            return o_tensor
        else:
            raise Exception("not support triton version")


class LlamaTransformerLayerInferINT4(LlamaTransformerLayerInfer):
    """
    Llama Model Inference using Triton W4A16 kernel.
    """

    def __init__(self, layer_num, tp_rank, world_size, network_config, mode=[], group_size=128):
        super().__init__(layer_num, tp_rank, world_size, network_config, mode)
        self.inter_dim_ = network_config['intermediate_size']
        self.q_group_size = group_size

    def _get_qkv(self, input, infer_state: LlamaInferStateInfo, layer_weight: LlamaTransformerLayerWeightQuantized, matmul_int4_func) -> torch.Tensor:
        qkv_output = matmul_int4_func(input.view(-1, self.embed_dim_),
                                      layer_weight.qkv_fused_weight,
                                      layer_weight.qkv_fused_weight_scale,
                                      layer_weight.qkv_fused_weight_zp,
                                      self.q_group_size)
        tp_hidden_dim = self.embed_dim_ // self.world_size_
        q = qkv_output[:, : tp_hidden_dim]
        k = qkv_output[:, tp_hidden_dim : tp_hidden_dim * 2]
        v = qkv_output[:, tp_hidden_dim * 2 :]
        rotary_emb_fwd(q.view(-1, self.tp_q_head_num_, self.head_dim_), infer_state.position_cos, infer_state.position_sin)
        cache_k_ = k.view(-1, self.tp_k_head_num_, self.head_dim_)
        rotary_emb_fwd(cache_k_, infer_state.position_cos, infer_state.position_sin)
        cache_v_ = v.view(-1, self.tp_v_head_num_, self.head_dim_)
        return q, cache_k_, cache_v_

    def _get_qkv_context(self, input, infer_state: LlamaInferStateInfo, layer_weight: LlamaTransformerLayerWeightQuantized) -> torch.Tensor:
        return self._get_qkv(input, infer_state, layer_weight, matmul_dequantize_int4_s1)

    def _get_qkv_decode(self, input, infer_state: LlamaInferStateInfo, layer_weight: LlamaTransformerLayerWeightQuantized) -> torch.Tensor:
        return self._get_qkv(input, infer_state, layer_weight, matmul_dequantize_int4_gptq)

    def _get_o_context(self, input, infer_state: LlamaInferStateInfo, layer_weight: LlamaTransformerLayerWeightQuantized) -> torch.Tensor:
        o_tensor = matmul_dequantize_int4_s1(input.view(-1, self.tp_o_head_num_ * self.head_dim_),
                                             layer_weight.o_weight_,
                                             layer_weight.o_weight_scale_,
                                             layer_weight.o_weight_zp_,
                                             self.q_group_size)
        return o_tensor

    def _get_o_decode(self, input, infer_state: LlamaInferStateInfo, layer_weight: LlamaTransformerLayerWeightQuantized) -> torch.Tensor:
        o_tensor = matmul_dequantize_int4_gptq(input.view(-1, self.tp_o_head_num_ * self.head_dim_),
                                             layer_weight.o_weight_,
                                             layer_weight.o_weight_scale_,
                                             layer_weight.o_weight_zp_,
                                             self.q_group_size)
        return o_tensor

    def _ffn(self, input, infer_state: LlamaInferStateInfo, layer_weight: LlamaTransformerLayerWeightQuantized, matmul_int4_func) -> torch.Tensor:
        gate_up_output = matmul_int4_func(input.view(-1, self.embed_dim_),
                                          layer_weight.gate_up_fused_weight,
                                          layer_weight.gate_up_fused_weight_scale,
                                          layer_weight.gate_up_fused_weight_zp,
                                          self.q_group_size)
        input = None
        tp_inter_dim = self.inter_dim_ // self.world_size_
        gate_up_output = gate_up_output.view(-1, 2, tp_inter_dim)
        torch.nn.functional.silu(gate_up_output[:, 0], inplace=True)
        ffn1_out = gate_up_output[:, 0] * gate_up_output[:, 1]
        gate_up_output = None
        ffn2_out = matmul_int4_func(ffn1_out,
                                    layer_weight.down_proj,
                                    layer_weight.down_proj_scale,
                                    layer_weight.down_proj_zp,
                                    self.q_group_size)
        ffn1_out = None
        return ffn2_out

    def _ffn_context(self, input, infer_state: LlamaInferStateInfo, layer_weight: LlamaTransformerLayerWeightQuantized) -> torch.Tensor:
        return self._ffn(input, infer_state, layer_weight, matmul_dequantize_int4_s1)

    def _ffn_decode(self, input, infer_state: LlamaInferStateInfo, layer_weight: LlamaTransformerLayerWeightQuantized) -> torch.Tensor:
        return self._ffn(input, infer_state, layer_weight, matmul_dequantize_int4_gptq)

    @mark_cost_time("trans context flash forward time cost")  # dont to remove this, will make performence down, did not know why
    def _context_attention(self, input_embding, infer_state: LlamaInferStateInfo, layer_weight):
        input1 = self._att_norm(input_embding, infer_state, layer_weight)
        self._pre_cache_kv(infer_state, layer_weight)
        q, cache_k, cache_v = self._get_qkv_context(input1, infer_state, layer_weight)
        input1 = None
        self._post_cache_kv(cache_k, cache_v, infer_state, layer_weight)
        o = self._context_attention_kernel(q, cache_k, cache_v, infer_state, layer_weight)
        q = None
        o = self._get_o_context(o, infer_state, layer_weight)
        if self.world_size_ > 1:
            dist.all_reduce(o, op=dist.ReduceOp.SUM, async_op=False)
        input_embding.add_(o.view(-1, self.embed_dim_))

    def _token_attention(self, input_embding, infer_state: LlamaInferStateInfo, layer_weight):
        input1 = self._att_norm(input_embding, infer_state, layer_weight)
        self._pre_cache_kv(infer_state, layer_weight)
        q, cache_k, cache_v = self._get_qkv_decode(input1, infer_state, layer_weight)
        input1 = None
        self._post_cache_kv(cache_k, cache_v, infer_state, layer_weight)
        o = self._token_attention_kernel(q, infer_state, layer_weight)
        q = None
        o = self._get_o_decode(o, infer_state, layer_weight)
        if self.world_size_ > 1:
            dist.all_reduce(o, op=dist.ReduceOp.SUM, async_op=False)
        input_embding.add_(o.view(-1, self.embed_dim_))

    @mark_cost_time("trans context ffn forward time cost")  # dont to remove this, will make performence down, did not know why
    def _context_ffn(self, input_embdings, infer_state: LlamaInferStateInfo, layer_weight):
        input1 = self._ffn_norm(input_embdings, infer_state, layer_weight)
        ffn_out = self._ffn_context(input1, infer_state, layer_weight)
        input1 = None
        if self.world_size_ > 1:
            dist.all_reduce(ffn_out, op=dist.ReduceOp.SUM, async_op=False)
        input_embdings.add_(ffn_out.view(-1, self.embed_dim_))

    def _token_ffn(self, input_embdings, infer_state: LlamaInferStateInfo, layer_weight):
        input1 = self._ffn_norm(input_embdings, infer_state, layer_weight)
        ffn_out = self._ffn_decode(input1, infer_state, layer_weight)
        input1 = None
        if self.world_size_ > 1:
            dist.all_reduce(ffn_out, op=dist.ReduceOp.SUM, async_op=False)
        input_embdings.add_(ffn_out.view(-1, self.embed_dim_))

    # Pre / post cache kv for fused weight.
    def _pre_cache_kv(self, infer_state: LlamaInferStateInfo, layer_weight)->Tuple[torch.Tensor, torch.Tensor]:
        '''
        Release kv buffer to save memory, since we allocate while kv projection.
        '''
        if infer_state.is_prefill:
            infer_state.prefill_key_buffer = None
            infer_state.prefill_value_buffer = None
        else:
            infer_state.decode_key_buffer = None
            infer_state.decode_value_buffer = None

    def _post_cache_kv(self, cache_k, cache_v, infer_state: LlamaInferStateInfo, layer_weight: LlamaTransformerLayerWeightQuantized):
        mem_manager = infer_state.mem_manager
        if infer_state.is_prefill:
            self._copy_kv_to_mem_cache(cache_k, cache_v, infer_state.prefill_mem_index, mem_manager)
        else:
            self._copy_kv_to_mem_cache(cache_k, cache_v, infer_state.decode_mem_index, mem_manager)


    def _token_decode_attention_normal(self, q, infer_state: LlamaInferStateInfo):
        '''
        We override token decode attention normal using triton kernel for 2.0.0 version,
        since there is bug in kernel for 2.1.0 version when using int4 weight.
        '''
        total_token_num = infer_state.total_token_num
        batch_size = infer_state.batch_size
        calcu_shape1 = (batch_size, self.tp_q_head_num_, self.head_dim_)
        att_m_tensor = torch.empty((self.tp_q_head_num_, total_token_num), dtype=q.dtype, device="cuda")

        token_att_fwd(q.view(calcu_shape1),
                      infer_state.mem_manager.key_buffer[self.layer_num_],
                      att_m_tensor,
                      infer_state.b_loc,
                      infer_state.b_start_loc,
                      infer_state.b_seq_len,
                      infer_state.max_len_in_batch)

        prob = torch.empty_like(att_m_tensor)
        token_softmax_fwd(att_m_tensor, infer_state.b_start_loc, infer_state.b_seq_len, prob, infer_state.max_len_in_batch)
        att_m_tensor = None

        o_tensor = torch.empty_like(q)

        token_att_fwd2(prob,
                       infer_state.mem_manager.value_buffer[self.layer_num_],
                       o_tensor.view(calcu_shape1),
                       infer_state.b_loc,
                       infer_state.b_start_loc,
                       infer_state.b_seq_len,
                       infer_state.max_len_in_batch)
        prob = None
        return o_tensor


class LlamaTransformerLayerInferINT4LMDeploy(LlamaTransformerLayerInfer):
    """
    Llama Model Inference using Triton W4A16 kernel.
    """

    def __init__(self, layer_num, tp_rank, world_size, network_config, mode=[], group_size=128):
        super().__init__(layer_num, tp_rank, world_size, network_config, mode)
        self.inter_dim_ = network_config['intermediate_size']
        self.q_group_size = group_size

    def _get_qkv(self, input, infer_state: LlamaInferStateInfo, layer_weight: LlamaTransformerLayerWeightQuantized, matmul_int4_func) -> torch.Tensor:
        qkv_output = matmul_int4_func(input.view(-1, self.embed_dim_),
                                      layer_weight.qkv_fused_weight,
                                      layer_weight.qkv_fused_weight_scale,
                                      layer_weight.qkv_fused_weight_zp,
                                      self.q_group_size)
        tp_hidden_dim = self.embed_dim_ // self.world_size_
        q = qkv_output[:, : tp_hidden_dim]
        k = qkv_output[:, tp_hidden_dim : tp_hidden_dim * 2]
        v = qkv_output[:, tp_hidden_dim * 2 :]
        rotary_emb_fwd(q.view(-1, self.tp_q_head_num_, self.head_dim_), infer_state.position_cos, infer_state.position_sin)
        cache_k_ = k.view(-1, self.tp_k_head_num_, self.head_dim_)
        rotary_emb_fwd(cache_k_, infer_state.position_cos, infer_state.position_sin)
        cache_v_ = v.view(-1, self.tp_v_head_num_, self.head_dim_)
        return q, cache_k_, cache_v_

    def _get_qkv_context(self, input, infer_state: LlamaInferStateInfo, layer_weight: LlamaTransformerLayerWeightQuantized) -> torch.Tensor:
        return self._get_qkv(input, infer_state, layer_weight, matmul_dequantize_int4_lmdeploy)

    def _get_qkv_decode(self, input, infer_state: LlamaInferStateInfo, layer_weight: LlamaTransformerLayerWeightQuantized) -> torch.Tensor:
        return self._get_qkv(input, infer_state, layer_weight, matmul_dequantize_int4_lmdeploy)

    def _get_o_context(self, input, infer_state: LlamaInferStateInfo, layer_weight: LlamaTransformerLayerWeightQuantized) -> torch.Tensor:
        o_tensor = matmul_dequantize_int4_lmdeploy(input.view(-1, self.tp_o_head_num_ * self.head_dim_),
                                             layer_weight.o_weight_,
                                             layer_weight.o_weight_scale_,
                                             layer_weight.o_weight_zp_,
                                             self.q_group_size)
        return o_tensor

    def _get_o_decode(self, input, infer_state: LlamaInferStateInfo, layer_weight: LlamaTransformerLayerWeightQuantized) -> torch.Tensor:
        o_tensor = matmul_dequantize_int4_lmdeploy(input.view(-1, self.tp_o_head_num_ * self.head_dim_),
                                             layer_weight.o_weight_,
                                             layer_weight.o_weight_scale_,
                                             layer_weight.o_weight_zp_,
                                             self.q_group_size)
        return o_tensor

    def _ffn(self, input, infer_state: LlamaInferStateInfo, layer_weight: LlamaTransformerLayerWeightQuantized, matmul_int4_func) -> torch.Tensor:
        gate_up_output = matmul_int4_func(input.view(-1, self.embed_dim_),
                                          layer_weight.gate_up_fused_weight,
                                          layer_weight.gate_up_fused_weight_scale,
                                          layer_weight.gate_up_fused_weight_zp,
                                          self.q_group_size)
        input = None
        tp_inter_dim = self.inter_dim_ // self.world_size_
        gate_up_output = gate_up_output.view(-1, 2, tp_inter_dim)
        torch.nn.functional.silu(gate_up_output[:, 0], inplace=True)
        ffn1_out = gate_up_output[:, 0] * gate_up_output[:, 1]
        gate_up_output = None
        ffn2_out = matmul_int4_func(ffn1_out,
                                    layer_weight.down_proj,
                                    layer_weight.down_proj_scale,
                                    layer_weight.down_proj_zp,
                                    self.q_group_size)
        ffn1_out = None
        return ffn2_out

    def _ffn_context(self, input, infer_state: LlamaInferStateInfo, layer_weight: LlamaTransformerLayerWeightQuantized) -> torch.Tensor:
        return self._ffn(input, infer_state, layer_weight, matmul_dequantize_int4_lmdeploy)

    def _ffn_decode(self, input, infer_state: LlamaInferStateInfo, layer_weight: LlamaTransformerLayerWeightQuantized) -> torch.Tensor:
        return self._ffn(input, infer_state, layer_weight, matmul_dequantize_int4_lmdeploy)

    @mark_cost_time("trans context flash forward time cost")  # dont to remove this, will make performence down, did not know why
    def _context_attention(self, input_embding, infer_state: LlamaInferStateInfo, layer_weight):
        input1 = self._att_norm(input_embding, infer_state, layer_weight)
        self._pre_cache_kv(infer_state, layer_weight)
        q, cache_k, cache_v = self._get_qkv_context(input1, infer_state, layer_weight)
        input1 = None
        self._post_cache_kv(cache_k, cache_v, infer_state, layer_weight)
        o = self._context_attention_kernel(q, cache_k, cache_v, infer_state, layer_weight)
        q = None
        o = self._get_o_context(o, infer_state, layer_weight)
        if self.world_size_ > 1:
            dist.all_reduce(o, op=dist.ReduceOp.SUM, async_op=False)
        input_embding.add_(o.view(-1, self.embed_dim_))

    def _token_attention(self, input_embding, infer_state: LlamaInferStateInfo, layer_weight):
        input1 = self._att_norm(input_embding, infer_state, layer_weight)
        self._pre_cache_kv(infer_state, layer_weight)
        q, cache_k, cache_v = self._get_qkv_decode(input1, infer_state, layer_weight)
        input1 = None
        self._post_cache_kv(cache_k, cache_v, infer_state, layer_weight)
        o = self._token_attention_kernel(q, infer_state, layer_weight)
        q = None
        o = self._get_o_decode(o, infer_state, layer_weight)
        if self.world_size_ > 1:
            dist.all_reduce(o, op=dist.ReduceOp.SUM, async_op=False)
        input_embding.add_(o.view(-1, self.embed_dim_))

    @mark_cost_time("trans context ffn forward time cost")  # dont to remove this, will make performence down, did not know why
    def _context_ffn(self, input_embdings, infer_state: LlamaInferStateInfo, layer_weight):
        input1 = self._ffn_norm(input_embdings, infer_state, layer_weight)
        ffn_out = self._ffn_context(input1, infer_state, layer_weight)
        input1 = None
        if self.world_size_ > 1:
            dist.all_reduce(ffn_out, op=dist.ReduceOp.SUM, async_op=False)
        input_embdings.add_(ffn_out.view(-1, self.embed_dim_))

    def _token_ffn(self, input_embdings, infer_state: LlamaInferStateInfo, layer_weight):
        input1 = self._ffn_norm(input_embdings, infer_state, layer_weight)
        ffn_out = self._ffn_decode(input1, infer_state, layer_weight)
        input1 = None
        if self.world_size_ > 1:
            dist.all_reduce(ffn_out, op=dist.ReduceOp.SUM, async_op=False)
        input_embdings.add_(ffn_out.view(-1, self.embed_dim_))

    # Pre / post cache kv for fused weight.
    def _pre_cache_kv(self, infer_state: LlamaInferStateInfo, layer_weight)->Tuple[torch.Tensor, torch.Tensor]:
        '''
        Release kv buffer to save memory, since we allocate while kv projection.
        '''
        if infer_state.is_prefill:
            infer_state.prefill_key_buffer = None
            infer_state.prefill_value_buffer = None
        else:
            infer_state.decode_key_buffer = None
            infer_state.decode_value_buffer = None

    def _post_cache_kv(self, cache_k, cache_v, infer_state: LlamaInferStateInfo, layer_weight: LlamaTransformerLayerWeightQuantized):
        mem_manager = infer_state.mem_manager
        if infer_state.is_prefill:
            self._copy_kv_to_mem_cache(cache_k, cache_v, infer_state.prefill_mem_index, mem_manager)
        else:
            self._copy_kv_to_mem_cache(cache_k, cache_v, infer_state.decode_mem_index, mem_manager)


    def _token_decode_attention_normal(self, q, infer_state: LlamaInferStateInfo):
        '''
        We override token decode attention normal using triton kernel for 2.0.0 version,
        since there is bug in kernel for 2.1.0 version when using int4 weight.
        '''
        total_token_num = infer_state.total_token_num
        batch_size = infer_state.batch_size
        calcu_shape1 = (batch_size, self.tp_q_head_num_, self.head_dim_)
        att_m_tensor = torch.empty((self.tp_q_head_num_, total_token_num), dtype=q.dtype, device="cuda")

        token_att_fwd(q.view(calcu_shape1),
                      infer_state.mem_manager.key_buffer[self.layer_num_],
                      att_m_tensor,
                      infer_state.b_loc,
                      infer_state.b_start_loc,
                      infer_state.b_seq_len,
                      infer_state.max_len_in_batch)

        prob = torch.empty_like(att_m_tensor)
        token_softmax_fwd(att_m_tensor, infer_state.b_start_loc, infer_state.b_seq_len, prob, infer_state.max_len_in_batch)
        att_m_tensor = None

        o_tensor = torch.empty_like(q)

        token_att_fwd2(prob,
                       infer_state.mem_manager.value_buffer[self.layer_num_],
                       o_tensor.view(calcu_shape1),
                       infer_state.b_loc,
                       infer_state.b_start_loc,
                       infer_state.b_seq_len,
                       infer_state.max_len_in_batch)
        prob = None
        return o_tensor

import math
import os
import numpy as np
import torch
from functools import partial

from lightllm.common.basemodel import TransformerLayerWeight
from lightllm.common.basemodel.triton_kernel.quantize_gemm_int8 import quantize_int8
from lightllm.common.basemodel.triton_kernel.dequantize_gemm_int4 import quantize_int4
from lightllm.common.basemodel.triton_kernel.dequantize_gemm_int4_lmdeploy import quantize_int4_lmdeploy


class LlamaTransformerLayerWeightQuantized(TransformerLayerWeight):
    def __init__(self, layer_num, tp_rank, world_size, data_type, network_config, mode=[], group_size=128):
        super().__init__(layer_num, tp_rank, world_size, data_type, network_config, mode)
        quantize_func_dict = {
            'int8weight': quantize_int8,
            'int4weight': partial(quantize_int4, group_size=group_size),
            'int4weight_lmdeploy': partial(quantize_int4_lmdeploy, group_size=group_size)
        }
        self.quantize_weight = None
        for item in mode:
            if item in quantize_func_dict:
                self.quantize_weight = quantize_func_dict[item]

    def load_hf_weights(self, weights):
        self._load_qkvo_weights(weights)
        self._load_ffn_weights(weights)

    def verify_load(self):
        errors = "weights load not ok"
        weights = [
            self.att_norm_weight_,
            self.qkv_fused_weight,
            self.o_weight_,
            self.ffn_norm_weight_,
            self.gate_up_fused_weight,
            self.down_proj
            ]
        for i in range(len(weights)):
            assert weights[i] is not None, "index:" + str(i) + " " + errors

    def _load_qkvo_weights(self, weights):
        # input layernorm params
        if f"model.layers.{self.layer_num_}.input_layernorm.weight" in weights:
            self.att_norm_weight_ = self._cuda(weights[f"model.layers.{self.layer_num_}.input_layernorm.weight"])

        n_embed = self.network_config_["hidden_size"]
        split_n_embed = n_embed // self.world_size_
        # q k v weights for llama
        if f"model.layers.{self.layer_num_}.self_attn.q_proj.weight" in weights:
            q_weight_ = weights[f"model.layers.{self.layer_num_}.self_attn.q_proj.weight"][split_n_embed *
                                                                                           self.tp_rank_: split_n_embed * (self.tp_rank_ + 1), :]
            q_weight_ = q_weight_.transpose(0, 1).to(self.data_type_)
            
            self.qkv_fused_weight = torch.empty(n_embed, split_n_embed * 3, dtype=self.data_type_, device='cpu')
            self.qkv_fused_weight[:, :split_n_embed] = q_weight_
        if f"model.layers.{self.layer_num_}.self_attn.k_proj.weight" in weights:
            k_weight_ = weights[f"model.layers.{self.layer_num_}.self_attn.k_proj.weight"][split_n_embed *
                                                                                           self.tp_rank_: split_n_embed * (self.tp_rank_ + 1), :]
            k_weight_ = k_weight_.transpose(0, 1).to(self.data_type_)
            self.qkv_fused_weight[:, split_n_embed:split_n_embed * 2] = k_weight_
        if f"model.layers.{self.layer_num_}.self_attn.v_proj.weight" in weights:
            v_weight_ = weights[f"model.layers.{self.layer_num_}.self_attn.v_proj.weight"][split_n_embed *
                                                                                           self.tp_rank_: split_n_embed * (self.tp_rank_ + 1), :]
            v_weight_ = v_weight_.transpose(0, 1).to(self.data_type_)
            self.qkv_fused_weight[:, split_n_embed * 2:split_n_embed * 3] = v_weight_
            self.qkv_fused_weight, self.qkv_fused_weight_scale, self.qkv_fused_weight_zp = \
                self.quantize_weight(self.qkv_fused_weight)
            self.qkv_fused_weight = self.qkv_fused_weight.cuda()
            self.qkv_fused_weight_scale = self.qkv_fused_weight_scale.to(self.data_type_).cuda() \
                if self.qkv_fused_weight_scale is not None else None
            self.qkv_fused_weight_zp = self.qkv_fused_weight_zp.cuda() \
                if self.qkv_fused_weight_zp is not None else None

        # attention output dense params
        if f"model.layers.{self.layer_num_}.self_attn.o_proj.weight" in weights:
            self.o_weight_ = weights[f"model.layers.{self.layer_num_}.self_attn.o_proj.weight"][:,
                                                                                                split_n_embed * self.tp_rank_: split_n_embed * (self.tp_rank_ + 1)]
            self.o_weight_, self.o_weight_scale_, self.o_weight_zp_ = \
                self.quantize_weight(self.o_weight_.transpose(0, 1))
            self.o_weight_ = self.o_weight_.cuda()
            self.o_weight_scale_ = self.o_weight_scale_.to(self.data_type_).cuda() if self.o_weight_scale_ is not None else None
            self.o_weight_zp_ = self.o_weight_zp_.cuda() if self.o_weight_zp_ is not None else None
    
    def _load_ffn_weights(self, weights):
        if f"model.layers.{self.layer_num_}.post_attention_layernorm.weight" in weights:
            self.ffn_norm_weight_ = self._cuda(weights[f"model.layers.{self.layer_num_}.post_attention_layernorm.weight"])
    
        n_embed = self.network_config_["hidden_size"]
        inter_size = self.network_config_['intermediate_size']
        split_inter_size = inter_size // self.world_size_

        if f"model.layers.{self.layer_num_}.mlp.gate_proj.weight" in weights:
            gate_proj = weights[f"model.layers.{self.layer_num_}.mlp.gate_proj.weight"][split_inter_size *
                                                                                        self.tp_rank_: split_inter_size * (self.tp_rank_ + 1), :]
            gate_proj = gate_proj.transpose(0, 1).to(self.data_type_)
            self.gate_up_fused_weight = torch.empty(n_embed, split_inter_size * 2, dtype=self.data_type_, device='cpu')
            self.gate_up_fused_weight[:, : split_inter_size] = gate_proj
        if f"model.layers.{self.layer_num_}.mlp.up_proj.weight" in weights:
            up_proj = weights[f"model.layers.{self.layer_num_}.mlp.up_proj.weight"][split_inter_size *
                                                                                    self.tp_rank_: split_inter_size * (self.tp_rank_ + 1), :]
            up_proj = up_proj.transpose(0, 1).to(self.data_type_)
            self.gate_up_fused_weight[:, split_inter_size : split_inter_size * 2] = up_proj
            self.gate_up_fused_weight, self.gate_up_fused_weight_scale, self.gate_up_fused_weight_zp = \
                self.quantize_weight(self.gate_up_fused_weight)
            self.gate_up_fused_weight = self.gate_up_fused_weight.cuda()
            self.gate_up_fused_weight_scale = self.gate_up_fused_weight_scale.to(self.data_type_).cuda() \
                if self.gate_up_fused_weight_scale is not None else None
            self.gate_up_fused_weight_zp = self.gate_up_fused_weight_zp.cuda() \
                if self.gate_up_fused_weight_zp is not None else None

        if f"model.layers.{self.layer_num_}.mlp.down_proj.weight" in weights:
            self.down_proj = weights[f"model.layers.{self.layer_num_}.mlp.down_proj.weight"][:,
                                                                                             split_inter_size * self.tp_rank_: split_inter_size * (self.tp_rank_ + 1)]
            self.down_proj, self.down_proj_scale, self.down_proj_zp = \
                self.quantize_weight(self.down_proj.transpose(0, 1))
            self.down_proj = self.down_proj.cuda()
            self.down_proj_scale = self.down_proj_scale.to(self.data_type_).cuda() if self.down_proj_scale is not None else None
            self.down_proj_zp = self.down_proj_zp.cuda() if self.down_proj_zp is not None else None


class LlamaTransformerLayerWeightOnlyQuantized(LlamaTransformerLayerWeightQuantized):
    def __init__(self, layer_num, tp_rank, world_size, data_type, network_config, mode=[], group_size=128):
        super().__init__(layer_num, tp_rank, world_size, data_type, network_config, mode)

    def _load_qkvo_weights(self, weights):
        # input layernorm params
        if f"model.layers.{self.layer_num_}.input_layernorm.weight" in weights:
            self.att_norm_weight_ = self._cuda(weights[f"model.layers.{self.layer_num_}.input_layernorm.weight"])

        # q k v weights for llama
        if f'model.layers.{self.layer_num_}.self_attn.qkv_proj.weight' in weights:
            self.qkv_fused_weight = weights[f'model.layers.{self.layer_num_}.self_attn.qkv_proj.weight']
            self.qkv_fused_weight_scale = weights.get(f'model.layers.{self.layer_num_}.self_attn.qkv_proj.weight.scale', None)
            self.qkv_fused_weight_zp = weights.get(f'model.layers.{self.layer_num_}.self_attn.qkv_proj.weight.zp', None)

            self.qkv_fused_weight = self.qkv_fused_weight.cuda()
            self.qkv_fused_weight_scale = self.qkv_fused_weight_scale.to(self.data_type_).cuda() \
                if self.qkv_fused_weight_scale is not None else None
            self.qkv_fused_weight_zp = self.qkv_fused_weight_zp.cuda() \
                if self.qkv_fused_weight_zp is not None else None

        # attention output dense params
        if f"model.layers.{self.layer_num_}.self_attn.o_proj.weight" in weights:
            self.o_weight_ = weights[f"model.layers.{self.layer_num_}.self_attn.o_proj.weight"]
            self.o_weight_scale_ = weights.get(f'model.layers.{self.layer_num_}.self_attn.o_proj.weight.scale', None)
            self.o_weight_zp_ = weights.get(f'model.layers.{self.layer_num_}.self_attn.o_proj.weight.zp', None)

            self.o_weight_ = self.o_weight_.cuda()
            self.o_weight_scale_ = self.o_weight_scale_.to(self.data_type_).cuda() if self.o_weight_scale_ is not None else None
            self.o_weight_zp_ = self.o_weight_zp_.cuda() if self.o_weight_zp_ is not None else None

    def _load_ffn_weights(self, weights):
        if f"model.layers.{self.layer_num_}.post_attention_layernorm.weight" in weights:
            self.ffn_norm_weight_ = self._cuda(weights[f"model.layers.{self.layer_num_}.post_attention_layernorm.weight"])

        if f"model.layers.{self.layer_num_}.mlp.gateup_proj.weight" in weights:
            self.gate_up_fused_weight = weights[f"model.layers.{self.layer_num_}.mlp.gateup_proj.weight"]
            self.gate_up_fused_weight_scale = weights.get(f'model.layers.{self.layer_num_}.mlp.gateup_proj.weight.scale', None)
            self.gate_up_fused_weight_zp = weights.get(f'model.layers.{self.layer_num_}.mlp.gateup_proj.weight.zp', None)

            self.gate_up_fused_weight = self.gate_up_fused_weight.cuda()
            self.gate_up_fused_weight_scale = self.gate_up_fused_weight_scale.to(self.data_type_).cuda() \
                if self.gate_up_fused_weight_scale is not None else None
            self.gate_up_fused_weight_zp = self.gate_up_fused_weight_zp.cuda() \
                if self.gate_up_fused_weight_zp is not None else None

        if f"model.layers.{self.layer_num_}.mlp.down_proj.weight" in weights:
            self.down_proj = weights[f"model.layers.{self.layer_num_}.mlp.down_proj.weight"]
            self.down_proj_scale = weights.get(f"model.layers.{self.layer_num_}.mlp.down_proj.weight.scale", None)
            self.down_proj_zp = weights.get(f"model.layers.{self.layer_num_}.mlp.down_proj.weight.zp", None)

            self.down_proj = self.down_proj.cuda()
            self.down_proj_scale = self.down_proj_scale.to(self.data_type_).cuda() if self.down_proj_scale is not None else None
            self.down_proj_zp = self.down_proj_zp.cuda() if self.down_proj_zp is not None else None

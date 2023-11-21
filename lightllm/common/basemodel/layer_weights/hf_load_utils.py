import torch
import os
import gc
from safetensors import safe_open

import mmap
import io
from collections import OrderedDict
import numpy as np
import struct


# from MTC, load 'xxx.flat' file by mmap, can save the memory
def load_flat_dict(fn):
    # Open the file in read mode
    with open(fn, "rb") as file_object:
        # Create a memory map of the file
        mmap_object = mmap.mmap(file_object.fileno(), 0, access=mmap.ACCESS_COPY)
        mem = memoryview(mmap_object)
        (head_offset,) = struct.unpack('<Q', mem[:8])
        data_mem = mem[8:head_offset]
        head_mem = mem[head_offset:]
        head_dct = torch.load(io.BytesIO(head_mem))
        res = OrderedDict()
        for k, (offset, sz, shape, dtype) in head_dct.items():
            mv = mem
            arr = torch.from_dlpack(np.frombuffer(mv[offset:offset+sz], dtype=dtype).reshape(shape))
            res[k] = arr
    return res


def load_func(file_, pre_post_layer=None, transformer_layer_list=None, weight_dir=None):
    # fix bug for 多线程加载的时候，每个线程内部的cuda device 会切回 0， 修改后来保证不会出现bug
    # !!! orin does not has dist package
    # import torch.distributed as dist
    # tp_rank = dist.get_rank()
    # torch.cuda.set_device(tp_rank)

    if file_.endswith(".flat"):
        weights = load_flat_dict(os.path.join(weight_dir, file_))
    elif file_.endswith(".safetensors"):
        weights = safe_open(os.path.join(weight_dir, file_), 'pt', 'cpu')
        weights = {k: weights.get_tensor(k) for k in weights.keys()}
    else:
        weights = torch.load(os.path.join(weight_dir, file_), 'cpu')

    if pre_post_layer is not None:
        pre_post_layer.load_hf_weights(weights)
    if transformer_layer_list is not None:
        for layer in transformer_layer_list:
            layer.load_hf_weights(weights)
    del weights
    gc.collect()


def load_hf_weights(data_type, weight_dir, pre_post_layer=None, transformer_layer_list=None, weight_dict=None):
    data_type = torch.float16 if data_type == 'fp16' else torch.float32
    if pre_post_layer is not None:
        assert pre_post_layer.data_type_ == data_type, "type is not right"
    if transformer_layer_list is not None:
        assert transformer_layer_list[0].data_type_ == data_type, "type is not right"
    if weight_dict:
        if pre_post_layer is not None:
            pre_post_layer.load_hf_weights(weight_dict)
        if transformer_layer_list is not None:
            for layer in transformer_layer_list:
                layer.load_hf_weights(weight_dict)
        del weight_dict
        return
    files = os.listdir(weight_dir)
    candidate_files = list(filter(lambda x : x.endswith('.flat'), files))
    if len(candidate_files) == 0:
        candidate_files = list(filter(lambda x : x.endswith('.safetensors'), files))
    if len(candidate_files) == 0:
        candidate_files = list(filter(lambda x : x.endswith('.bin'), files))
    assert len(candidate_files) != 0, "can only support pytorch tensor and safetensors format for weights."
    from functools import partial
    from multiprocessing.pool import ThreadPool as Pool
    partial_func = partial(load_func, pre_post_layer=pre_post_layer, transformer_layer_list=transformer_layer_list, weight_dir=weight_dir)  # noqa
    worker = os.environ.get('LOADWORKER', 1)
    with Pool(worker) as p:
        _ = p.map(partial_func, candidate_files)
    return

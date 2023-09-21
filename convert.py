import json
import logging
import sys
import time
import os
from typing import Dict, List, Optional, Tuple

import torch

from .model.nn import Net
from .model.ps import Ps

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s',
    level=logging.INFO)

CLICK_SEQUENCE_LENGTH = 256
SPARSE_FEATURE_LIST = [
    'uid', 'os', 'city_id', 'county_fibs', 'dma_code', 'state_code',
    'y_id', 'y_domain', 'y_cate1', 'y_cate2', 'y_cate3', 'y_chn', 'y_seg_title', 'y_pid', 'y_zip_code',
    "local_x_id", "local_x_domain", "local_x_cate1", "local_x_cate2", "local_x_cate3", "local_x_chn", "local_x_seg_title", "local_x_pid",
    "nonlocal_x_id", "nonlocal_x_domain", "nonlocal_x_cate1", "nonlocal_x_cate2", "nonlocal_x_cate3", "nonlocal_x_chn", "nonlocal_x_seg_title", "nonlocal_x_pid",
]

feature_info = [
    {"name": "hashed_uid", "data_type": "tensor", "size": 0, "tensor_properties": {"shape": [1], "dtype": "INT64", "embedding_size": 9999992}},
    {"name": "os", "data_type": "tensor", "size": 0, "tensor_properties": {"shape": [1], "dtype": "INT64", "embedding_size": 3}},
    {"name": "on_boarding_days", "data_type": "tensor", "size": 0, "tensor_properties": {"shape": [1], "dtype": "INT64", "embedding_size": 11}},
    {"name": "city_id", "data_type": "tensor", "size": 0, "tensor_properties": {"shape": [1], "dtype": "INT64", "embedding_size": 30000}},
    {"name": "county_fibs", "data_type": "tensor", "size": 0, "tensor_properties": {"shape": [1], "dtype": "INT64", "embedding_size": 60000}},
    {"name": "state_code", "data_type": "tensor", "size": 0, "tensor_properties": {"shape": [1], "dtype": "INT64", "embedding_size": 60}},
    {"name": "dma_code", "data_type": "tensor", "size": 0, "tensor_properties": {"shape": [1], "dtype": "INT64", "embedding_size": 300}},
    
    {"name": "check", "data_type": "tensor", "size": 0, "tensor_properties": {"shape": [-1], "dtype": "INT64", "embedding_size": 21}},
    {"name": "click", "data_type": "tensor", "size": 0, "tensor_properties": {"shape": [-1], "dtype": "INT64", "embedding_size": 21}},
    {"name": "ctr", "data_type": "tensor", "size": 0, "tensor_properties": {"shape": [-1], "dtype": "FLOAT32"}},
    {"name": "level_score", "data_type": "tensor", "size": 0, "tensor_properties": {"shape": [-1], "dtype": "FLOAT32"}},
    {"name": "avg_level_score", "data_type": "tensor", "size": 0, "tensor_properties": {"shape": [-1], "dtype": "FLOAT32"}},
    {"name": "zip_code", "data_type": "tensor", "size": 0, "tensor_properties": {"shape": [1], "dtype": "INT64", "embedding_size": 200000}},

    {"name": "local_xs", "data_type": "tensor", "size": 0, "tensor_properties": {"shape": [CLICK_SEQUENCE_LENGTH], "dtype": "INT64", "embedding_size": 5000088}},    
    {"name": "local_rdid_mask", "data_type": "tensor", "size": 0, "tensor_properties": {"shape": [CLICK_SEQUENCE_LENGTH], "dtype": "INT64", "embedding_size": 2}},    
    {"name": "local_x_domain", "data_type": "tensor", "size": 0, "tensor_properties": {"shape": [CLICK_SEQUENCE_LENGTH], "dtype": "INT64", "embedding_size": 30000}},    
    {"name": "local_x_cate", "data_type": "tensor", "size": 0, "tensor_properties": {"shape": [CLICK_SEQUENCE_LENGTH, 9], "dtype": "INT64", "embedding_size": 800}},    
    {"name": "local_x_chn", "data_type": "tensor", "size": 0, "tensor_properties": {"shape": [CLICK_SEQUENCE_LENGTH, 5], "dtype": "INT64", "embedding_size": 150000}},    
    {"name": "local_x_seg_title", "data_type": "tensor", "size": 0, "tensor_properties": {"shape": [CLICK_SEQUENCE_LENGTH, 10], "dtype": "INT64", "embedding_size": 100000}},
    {"name": "local_x_pid", "data_type": "tensor", "size": 0, "tensor_properties": {"shape": [CLICK_SEQUENCE_LENGTH, 3], "dtype": "INT64", "embedding_size": 100000}},    
    {"name": "local_x_semantic_emb", "data_type": "tensor", "size": 0, "tensor_properties": {"shape": [CLICK_SEQUENCE_LENGTH, 50], "dtype": "FLOAT32"}},

    {"name": "nonlocal_xs", "data_type": "tensor", "size": 0, "tensor_properties": {"shape": [CLICK_SEQUENCE_LENGTH], "dtype": "INT64", "embedding_size": 5000088}},    
    {"name": "nonlocal_rdid_mask", "data_type": "tensor", "size": 0, "tensor_properties": {"shape": [CLICK_SEQUENCE_LENGTH], "dtype": "INT64", "embedding_size": 2}},    
    {"name": "nonlocal_x_domain", "data_type": "tensor", "size": 0, "tensor_properties": {"shape": [CLICK_SEQUENCE_LENGTH], "dtype": "INT64", "embedding_size": 30000}},    
    {"name": "nonlocal_x_cate", "data_type": "tensor", "size": 0, "tensor_properties": {"shape": [CLICK_SEQUENCE_LENGTH, 9], "dtype": "INT64", "embedding_size": 800}},    
    {"name": "nonlocal_x_chn", "data_type": "tensor", "size": 0, "tensor_properties": {"shape": [CLICK_SEQUENCE_LENGTH, 5], "dtype": "INT64", "embedding_size": 150000}},    
    {"name": "nonlocal_x_seg_title", "data_type": "tensor", "size": 0, "tensor_properties": {"shape": [CLICK_SEQUENCE_LENGTH, 10], "dtype": "INT64", "embedding_size": 100000}},
    {"name": "nonlocal_x_pid", "data_type": "tensor", "size": 0, "tensor_properties": {"shape": [CLICK_SEQUENCE_LENGTH, 3], "dtype": "INT64", "embedding_size": 100000}},    
    {"name": "nonlocal_x_semantic_emb", "data_type": "tensor", "size": 0, "tensor_properties": {"shape": [CLICK_SEQUENCE_LENGTH, 50], "dtype": "FLOAT32"}},
    
    {"name": "ys", "data_type": "tensor", "size": 0, "tensor_properties": {"shape": [-1], "dtype": "INT64", "embedding_size": 5000088}},    
    {"name": "y_domain", "data_type": "tensor", "size": 0, "tensor_properties": {"shape": [-1], "dtype": "INT64", "embedding_size": 30000}},    
    {"name": "y_cate", "data_type": "tensor", "size": 0, "tensor_properties": {"shape": [-1, 9], "dtype": "INT64", "embedding_size": 800}},    
    {"name": "y_chn", "data_type": "tensor", "size": 0, "tensor_properties": {"shape": [-1, 5], "dtype": "INT64", "embedding_size": 150000}},    
    {"name": "y_seg_title", "data_type": "tensor", "size": 0, "tensor_properties": {"shape": [-1, 10], "dtype": "INT64", "embedding_size": 100000}},    
    {"name": "y_pid", "data_type": "tensor", "size": 0, "tensor_properties": {"shape": [-1, 3], "dtype": "INT64", "embedding_size": 100000}},    
    {"name": "y_semantic_emb", "data_type": "tensor", "size": 0, "tensor_properties": {"shape": [-1, 50], "dtype": "FLOAT32"}},
]

class Wrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, device_type: str='cpu'):
        super().__init__()
        self.model = model
        self.device_type = device_type
        self.debug_flag = "__debug_flag__"

    def forward(
        self,
        dense_feature_data: Dict[str, torch.Tensor],
        sparse_feature_data: List[Tuple[str, List[List[str]]]],
        embedding_feature_data: Dict[str, torch.Tensor],
        tensor_feature_data: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        is_debug = self.debug_flag in dense_feature_data
        outs = self.model(tensor_feature_data)
        if len(outs.shape) == 1:  # when predict docid size is 1
            outs = outs.unsqueeze(0)

        result = {"predictions": torch.exp(outs[:, 1])}
        if is_debug:
            for k, v in tensor_feature_data.items():
                result['__debug__' + k] = v
        return result

def generate_random_batches(num_batches: int = 10, x_num: int = 80, y_num: int = 25):
    batches = [{
        "local_xs": torch.randint(0, 100, (x_num, )),
        "local_x_domain": torch.randint(0, 100, (x_num, )),
        "local_x_cate": torch.randint(0, 100, (x_num, 5)), # 每个batch x个样本 每个样本5个cate
        "local_x_chn": torch.randint(0, 100, (x_num, 5)),
        "local_x_seg_title": torch.randint(0, 100, (x_num, 5)),
        "local_x_pid": torch.randint(0, 100, (x_num, 5)),
        "local_x_semantic_emb": torch.rand(x_num, 50),
        "local_rdid_mask": torch.randint(0, 100, (y_num, x_num)),
        "nonlocal_rdid_mask": torch.randint(0, 100, (y_num, x_num)),
        "nonlocal_xs": torch.randint(0, 100, (x_num, )),
        "nonlocal_x_domain": torch.randint(0, 100, (x_num, )),
        "nonlocal_x_cate": torch.randint(0, 100, (x_num, 5)), 
        "nonlocal_x_chn": torch.randint(0, 100, (x_num, 5)),
        "nonlocal_x_seg_title": torch.randint(0, 100, (x_num, 5)),
        "nonlocal_x_pid": torch.randint(0, 100, (x_num, 5)),
        "nonlocal_x_semantic_emb": torch.rand(x_num, 50),
        "ys": torch.randint(0, 100, (y_num, )),
        "y_domain": torch.randint(0, 100, (y_num, )),
        "y_cate": torch.randint(0, 100, (y_num, 5)), 
        "y_chn": torch.randint(0, 100, (y_num, 5)),
        "y_seg_title": torch.randint(0, 100, (y_num, 5)),
        "y_pid": torch.randint(0, 100, (y_num, 5)),
        "y_semantic_emb": torch.rand(y_num, 50),
        "os": torch.randint(0, 2, (1, )),
        "uid": torch.randint(0, 100, (1, )),
        "hashed_uid": torch.randint(0, 100, (1, )),
        "zip_code": torch.randint(0, 100, (y_num, )),

        "level_score": torch.rand(y_num),
        "on_boarding_days": torch.randint(0, 5, (1,)),
        "county_fibs": torch.randint(0, 100, (1,)),
        "state_code": torch.randint(0, 50, (1,)),
        "city_id": torch.randint(0, 100, (1,)),
        "dma_code": torch.randint(0, 50, (1,)),
        "check": torch.randint(0, 20000, (y_num, )),
        "click": torch.randint(0, 10000, (y_num, )),
        "ctr": torch.rand(y_num),
        "avg_level_score": torch.rand(y_num),
    } for _ in range(num_batches)]  
    return batches


def atomic_write_model(model, output_path, extra_data):
    if os.path.exists(output_path):
        os.remove(output_path)
        logging.info(f"Deleted old model {output_path}")

    temp_path = output_path + ".__temp__"
    logging.info(f"Writing model to {temp_path}")
    torch.jit.save(model, temp_path, _extra_files=extra_data)
    os.rename(temp_path, output_path)
    logging.info(f"Renamed model to {output_path}")

def generate_random_tensor(batch_size, info):
    raw_shape = info["tensor_properties"]["shape"]
    shape = raw_shape
    if raw_shape[0] == -1:
        shape = [batch_size] + raw_shape[1:]
    if info["tensor_properties"]["dtype"].startswith("INT"):
        return torch.randint(0, info["tensor_properties"]["embedding_size"], shape)
    else:
        return torch.randn(shape)

def gen_mock_input(feature_info, enable_sparse_to_tensor, bs):
    # mock数据
    sparse_data = [(f['name'], [[f['options'][0]] * f['size'] for _ in range(bs)]) for f in feature_info if f['data_type'] == 'string']
    embedding_data = {f['name']: torch.rand(bs, f["size"]) for f in feature_info if f['data_type'] == 'double' and f["size"] > 1}
    dense_data = {f['name']: torch.rand(bs) for f in feature_info if f['data_type'] == 'double' and f["size"] == 1}
    tensor_data = {f['name']: generate_random_tensor(bs, f) for f in feature_info if f['data_type'] == 'tensor'}
    return sparse_data, dense_data, embedding_data, tensor_data

def generate_test_features(mock=True, batch_size=50):
    sparse_data, dense_data, embedding_data, tensor_data = list(), dict(), dict(), dict()

    enable_sparse_to_tensor = True
    data = None
    if mock:
        sparse_data, dense_data, embedding_data, tensor_data = gen_mock_input(feature_info, enable_sparse_to_tensor, batch_size)

    return sparse_data, dense_data, embedding_data, tensor_data, data


def push_model(model_output_dir, model_name, expected_batch_size, device_type):
    models = {
        "push_net": Net(),
        "ps": Ps(),
    }
    logging.info("Created model")

    net = models['push_net']
    net.ps = models['ps']
    
    num_batches = 10
    sequence_length = CLICK_SEQUENCE_LENGTH
    batches = generate_random_batches(num_batches, sequence_length, expected_batch_size)
    logging.info(f"Generated random data with {num_batches} batches and batch size {expected_batch_size} and sequence length {sequence_length}")

    traced_model = torch.jit.trace(net, [batches[0]], check_inputs=[(b, ) for b in batches])
    final_model = torch.jit.script(Wrapper(traced_model, device_type))
    logging.info("Torchscripted model")

    if device_type == 'cpu':
        sparse_data, dense_data, embedding_data, tensor_data, origin_data = generate_test_features(expected_batch_size)
        output = final_model(dense_data, sparse_data, embedding_data, tensor_data)

    for param in final_model.parameters():
        param.requires_grad = False

    extra_data = {
        "metadata.json": json.dumps(
            {
                "feature_info": feature_info,
                "version_info": {
                    "expected_batch_size": expected_batch_size
                }
            })
    }
    
    output_file = os.path.join(model_output_dir, f"{model_name}.pt") 

    atomic_write_model(final_model, output_file, extra_data)

    logging.info("Finished converting model")



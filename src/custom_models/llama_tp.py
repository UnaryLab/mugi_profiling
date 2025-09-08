import torch
import torch.nn as nn
import torch.distributed as dist

def init_dist():
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())

def patch_embedding(model):
    print(model.model)
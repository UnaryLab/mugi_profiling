import torch
import torch.nn as nn
import torch.distributed as dist

def init_dist():
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())

def get_dist():


def patch_embedding()
import torch
import torch.nn as nn
import torch.distributed as dist

def init_dist():
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())
    print("Current CUDA device index:", torch.cuda.current_device())
    return dist.get_world_size()
    
def patch_embedding(model, world_size):
    print(model.model.embed_tokens)
    embed_tokens = model.model.embed_tokens
    vocab_size, embed_dim = embed_tokens.weight.shape
    exit()
import torch
import torch.nn as nn
import torch.distributed as dist

from src.custom_models.parallel_modules import ColumnParallelRMSNorm

def init_dist():
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())
    return dist.get_world_size(), dist.get_rank()
    
def patch_embedding(model, world_size, rank):
    embed_tokens = model.model.embed_tokens
    vocab_size, embed_dim = embed_tokens.weight.shape

    hidden_per_gpu = embed_dim // world_size
    start = rank * hidden_per_gpu
    end = (rank + 1) * hidden_per_gpu

    local_weight = embed_tokens.weight[:, start:end].contiguous()
    local_embed = nn.Embedding(vocab_size, hidden_per_gpu)
    with torch.no_grad():
        local_embed.weight.copy_(local_weight)
    local_embed = local_embed.cuda()

    model.model.embed_tokens = local_embed

def patch_rmsnorm(model, world_size, rank):
    rmsnorm = model.model.norm
    hidden_size = rmsnorm.weight.shape[0]
    eps = model.config.rms_norm_eps

    print('here')
    print(hidden_size, eps)

    hidden_per_gpu = hidden_size // world_size
    start = rank * hidden_per_gpu
    end = (rank + 1) * hidden_per_gpu

    local_weight = rmsnorm.weight[start:end].contiguous()
    local_rmsnorm = ColumnParallelRMSNorm(world_size, hidden_size, eps)
    with torch.no_grad():
        local_rmsnorm.weight.copy_(local_weight)
    local_rmsnorm = local_rmsnorm.cuda()

    model.model.norm = local_rmsnorm
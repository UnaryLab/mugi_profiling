import torch
import torch.nn as nn
import torch.distributed as dist

from src.custom_models.parallel_modules import ColumnParallelRMSNorm, ColumnParallelLinear, ColumnParallelEmbedding

def init_dist():
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())
    return dist.get_world_size(), dist.get_rank()
    
def patch_embedding(embed_tokens, vocab_size, hidden_size, padding_idx, world_size):
    local_embed = ColumnParallelEmbedding(vocab_size, hidden_size, padding_idx, world_size)
    with torch.no_grad():
        local_embed.embed_tokens.weight.copy_(embed_tokens.weight)
    local_embed = local_embed.cuda()
    return local_embed

def patch_rmsnorm(rmsnorm, eps, world_size):
    hidden_size = rmsnorm.weight.shape[0]
    local_rmsnorm = ColumnParallelRMSNorm(world_size, hidden_size, eps)
    with torch.no_grad():
        local_rmsnorm.weight.copy_(rmsnorm.weight)
    local_rmsnorm = local_rmsnorm.cuda()
    return local_rmsnorm

# def patch_embedding(embed_tokens, world_size, rank):
#     vocab_size, embed_dim = embed_tokens.weight.shape

#     hidden_per_gpu = embed_dim // world_size
#     start = rank * hidden_per_gpu
#     end = (rank + 1) * hidden_per_gpu

#     local_weight = embed_tokens.weight[:, start:end].contiguous()
#     local_embed = nn.Embedding(vocab_size, hidden_per_gpu)
#     with torch.no_grad():
#         local_embed.weight.copy_(local_weight)
#     local_embed = local_embed.cuda()

#     return local_embed

# def patch_rmsnorm(rmsnorm, eps, world_size, rank):
#     hidden_size = rmsnorm.weight.shape[0]

#     hidden_per_gpu = hidden_size // world_size
#     start = rank * hidden_per_gpu
#     end = (rank + 1) * hidden_per_gpu

#     local_weight = rmsnorm.weight[start:end].contiguous()
#     local_rmsnorm = ColumnParallelRMSNorm(world_size, hidden_size, eps)
#     with torch.no_grad():
#         local_rmsnorm.weight.copy_(local_weight)
#     local_rmsnorm = local_rmsnorm.cuda()

#     return local_rmsnorm

def patch_linear(linear, bias, world_size, rank):
    n, k = linear.weight.shape

    n_per_gpu = n // world_size
    start = rank * n_per_gpu
    end = (rank + 1) * n_per_gpu

    local_weight = linear.weight[start:end, :].contiguous()
    local_linear = nn.Linear(k, n_per_gpu, bias=bias, dtype=linear.weight.dtype)
    
    with torch.no_grad():
        local_linear.weight.copy_(local_weight)
    local_linear = local_linear.cuda()

    return local_linear
    
def patch_output_linear(linear, bias, world_size, rank):
    return ColumnParallelLinear(linear.in_features, linear.out_features, linear.weight, bias, world_size).cuda()
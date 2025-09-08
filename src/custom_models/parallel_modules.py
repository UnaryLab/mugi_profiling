import torch
import torch.nn as nn
import torch.distributed as dist

class ColumnParallelRMSNorm(nn.Module):
    def __init__(self, world_size, hidden_size, eps):
        super().__init__()
        self.world_size = world_size
        self.eps = eps
        self.hidden_per_gpu = hidden_size // self.world_size
        self.weight = nn.Parameter(torch.ones(self.hidden_per_gpu))

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)

        local_sum_sq = hidden_states.pow(2).sum(-1, keepdim=True)
        dist.all_reduce(local_sum_sq, op=dist.ReduceOp.SUM, group=None)
        variance = local_sum_sq / (self.hidden_per_gpu * self.world_size)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return self.weight * hidden_states.to(input_dtype)
import torch
import torch.nn as nn
import torch.distributed as dist

class ColumnParallelRMSNorm(nn.Module):
    def __init__(self, world_size, hidden_size, eps, dtype):
        super().__init__()
        self.world_size = world_size
        self.eps = eps
        self.hidden_size = hidden_size
        self.weight = nn.Parameter(torch.ones(self.hidden_size)).to(dtype).cuda()

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)

        local_sum_sq = hidden_states.pow(2).sum(-1, keepdim=True)
        dist.all_reduce(local_sum_sq, op=dist.ReduceOp.SUM, group=None)
        variance = local_sum_sq / (self.hidden_size * self.world_size)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return self.weight * hidden_states.to(input_dtype)
    
class ColumnParallelEmbedding(nn.Module):
    def __init__(self, vocab_size, hidden_size, padding_idx, dtype):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size, padding_idx).to(dtype).cuda()

    def forward(self, inputs):
        return self.embed_tokens(inputs)

class ColumnParallelLinear(nn.Module):
    def __init__(self, in_features, out_features, weights, bias, world_size):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.world_size = world_size
        self.out_per_gpu = out_features // world_size

        self.output = nn.Linear(in_features, out_features, bias=bias).to(weights.dtype).cuda()
        with torch.no_grad():
            self.output.weight.copy_(weights)
    
    def forward(self, input):
        gathered = [torch.empty_like(input) for _ in range(self.world_size)]
        dist.all_gather(gathered, input)
        full_input = torch.cat(gathered, dim=-1)
        return self.output(full_input)
    
class OutputLayer(nn.Module):
    def __init__(self, in_features, out_features, weights, bias, world_size, rank):
        super().__init__()
        if rank == 0:
            self.in_features = in_features
            self.out_features = out_features
            self.world_size = world_size
            self.rank = rank

            self.output = nn.Linear(in_features, out_features, bias=bias).to(weights.dtype).cuda()
            with torch.no_grad():
                self.output.weight.copy_(weights)
        else:
            self.rank = rank
    
    def forward(self, input):
        if self.rank == 0:
            # gathered = [torch.empty_like(input) for _ in range(self.world_size)]
            # dist.all_gather(gathered, input)
            # full_input = torch.cat(gathered, dim=-1)
            output = self.output(input)
            return output
        return input
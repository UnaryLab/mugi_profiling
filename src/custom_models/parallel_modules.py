import torch
import torch.nn as nn
import torch.distributed as dist

class ColumnParallelRMSNorm(nn.Module):
    def __init__(self, world_size, hidden_size, eps):
        super().__init__()
        self.world_size = world_size
        self.eps = eps
        self.hidden_size = hidden_size
        self.weight = nn.Parameter(torch.ones(self.hidden_size))

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)

        local_sum_sq = hidden_states.pow(2).sum(-1, keepdim=True)
        dist.all_reduce(local_sum_sq, op=dist.ReduceOp.SUM, group=None)
        variance = local_sum_sq / (self.hidden_size * self.world_size)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return self.weight * hidden_states.to(input_dtype)
    
class ColumnParallelEmbedding(nn.Module):
    def __init__(self, vocab_size, hidden_size, padding_idx, world_size):
        super().__init__()
        self.world_size = world_size
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size, padding_idx).cuda()

    def forward(self, inputs):
        print('inputs', inputs.shape, inputs.device, inputs.dtype)
        print('embedding', self.embed_tokens.weight.shape, self.embed_tokens.weight.device, self.embed_tokens.weight.dtype)
        inputs = self.embed_tokens(inputs)
        print('inputs', inputs.shape, inputs.device, inputs.dtype)
        print('embedding', self.embed_tokens.weight.shape, self.embed_tokens.weight.device, self.embed_tokens.weight.dtype)
        exit()
        return inputs

class ColumnParallelLinear(nn.Module):
    def __init__(self, in_features, out_features, weights, bias, world_size):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.world_size = world_size
        self.out_per_gpu = out_features // world_size

        self.output = nn.Linear(in_features, out_features, bias=bias)
        with torch.no_grad():
            self.output.weight.copy_(weights)
    
    def forward(self, input):
        gathered = [torch.empty_like(input) for _ in range(self.world_size)]
        dist.all_gather(gathered, input)
        full_input = torch.cat(gathered, dim=-1)
        return self.output(full_input)
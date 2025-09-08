import torch
import torch.nn as nn
import torch.distributed as dist

class ColumnParallelModule(nn.Module):
    def __init__(self, process_group=None):
        super().__init__()
        self.process_group = process_group or dist.group.WORLD
        self.world_size = dist.get_world_size(self.process_group)
        self.rank = dist.get_rank(self.process_group)
        self.device = torch.device(f'cuda:{self.rank}')

class ColumnParallelEmbedding(nn.Module):
    def __init__(self, num_embeddings, hidden_size, process_group, padding_idx=None):
        super().__init__()
        self.world_size = dist.get_world_size(process_group)
        self.rank = dist.get_rank(process_group)
        self.hidden_per_gpu = hidden_size // self.world_size

        self.weight = nn.Parameter(
            torch.empty(num_embeddings, self.hidden_per_gpu, device=f'cuda:{self.rank}')
        )
        self.padding_idx = padding_idx

        nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def forward(self, input_ids):
        """
        input_ids: LongTensor of shape [batch, seq_len]
        Returns: [batch, seq_len, hidden_per_gpu]
        """
        # Simply lookup rows in this GPU's slice
        local_embed = self.weight[input_ids]  # [batch, seq_len, hidden_per_gpu]
        return local_embed
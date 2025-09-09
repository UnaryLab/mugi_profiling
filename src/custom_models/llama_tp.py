import torch
import torch.nn as nn
import torch.distributed as dist

from src.custom_models.parallel_modules import ColumnParallelRMSNorm, ColumnParallelLinear, ColumnParallelEmbedding

from typing import Optional, Union
from typing_extensions import Unpack
from transformers.cache_utils import Cache
from transformers.models.llama.modeling_llama import KwargsForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast

def init_dist():
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())
    return dist.get_world_size(), dist.get_rank()
    
def patch_embedding(embed_tokens, vocab_size, hidden_size, padding_idx):
    dtype = embed_tokens.weight.dtype
    local_embed = ColumnParallelEmbedding(vocab_size, hidden_size, padding_idx, dtype)
    with torch.no_grad():
        local_embed.embed_tokens.weight.copy_(embed_tokens.weight)
    local_embed = local_embed.cuda()
    return local_embed

def patch_rmsnorm(rmsnorm, eps, world_size):
    dtype = rmsnorm.weight.dtype
    hidden_size = rmsnorm.weight.shape[0]
    local_rmsnorm = ColumnParallelRMSNorm(world_size, hidden_size, eps, dtype)
    with torch.no_grad():
        local_rmsnorm.weight = rmsnorm.weight.copy_(rmsnorm.weight)
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

def patch_linear(linear, bias, world_size, rank, dim):
    n, k = linear.weight.shape

    if dim == 0:
        k_per_gpu = k // world_size
        start = rank * k_per_gpu
        end = (rank + 1) * k_per_gpu

        local_weight = linear.weight[:, start:end].contiguous()
        local_linear = nn.Linear(k_per_gpu, n, bias=bias, dtype=linear.weight.dtype)
    else:
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

def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[KwargsForCausalLM],
    ) -> CausalLMOutputWithPast:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state

        # Accumulate all slices from all ranks inplace
        if dist.get_rank() == 0:
            dist.all_reduce(hidden_states, op=dist.ReduceOp.SUM)

            # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
            slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
            logits = self.lm_head(hidden_states[:, slice_indices, :])

            loss = None
            if labels is not None:
                loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

            return CausalLMOutputWithPast(
                loss=loss,
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )
        else:
            return None
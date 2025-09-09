
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.inference_classes.inference_class import InferenceModel
from src.custom_nonlinear.custom_eager import LlamaEager
from src.custom_nonlinear.custom_forward import llama_forward

from src.custom_models.llama_tp import init_dist, patch_embedding, patch_rmsnorm, patch_linear, patch_output_linear

import torch
import types

class LlamaInference(InferenceModel):
    def __init__(self, model_dict, nonlinear_dict, parameter_dict, device):
        super().__init__(model_dict, nonlinear_dict, parameter_dict, device)
        self.world_size, self.rank = init_dist()

    def batch_dataset(self):
        batched_data = []
        for i in range(0, self.n_samples, self.batch_size):
            if i + self.batch_size > self.n_samples:
                batch = self.inputs[i:]
            else:
                batch = self.inputs[i:i + self.batch_size]

            batch_max_len = max(ex['input_ids'].shape[1] for ex in batch)
            padded_batch = []
            for ex in batch:
                input_ids = torch.nn.utils.rnn.pad_sequence(
                    [ex['input_ids'].squeeze(0)[:batch_max_len]],
                    batch_first=True,
                    padding_value=self.tokenizer.pad_token_id
                ).squeeze(0)

                attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

                padded_batch.append({
                    'input_ids': input_ids,
                    'attention_mask': attention_mask
                })
            batch = padded_batch
            batched_data.append(batch)

        self.inputs = batched_data

    def compute_metric(self, total_loss, num_batches):
        return self.compute_perplexity(total_loss, num_batches)
    
    def load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch.float16, attn_implementation='eager', device_map={"":"cuda:0", "":"cuda:1"}, use_cache=False)
        self.max_length = self.model.config.max_position_embeddings

        if self.max_length > 4096:
            self.max_length = 4096

        self.tp_patch()

    def patch_layers(self, attention_class, ffn_class, path):
        for i, layer in enumerate(self.model.model.layers):
                layer_device = next(layer.parameters()).device

                self.append_nonlinear_list(attention_class=attention_class,
                                           ffn_class=ffn_class,
                                           layer=i,
                                           device=layer_device,
                                           path=path,
                                           profiling_dims=self.profiling_dims)

                eager_attn_fn = LlamaEager(nonlinear_object=self.attention_objects[i])
                forward = llama_forward(eager_attn_fn)
                
                layer.self_attn.forward = types.MethodType(forward, layer.self_attn)
                layer.mlp.act_fn = self.ffn_objects[i]


    def process_dataset(self):
        self.inputs = []
        for example in self.dataset:
            if len(self.inputs) >= self.n_samples:
                break
            if len(example["text"]) > self.max_length:
                tokenized_example = self.tokenizer(
                    example["text"],
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt"
                )
                if tokenized_example['input_ids'].shape[-1] >= self.max_length:
                    self.inputs.append(tokenized_example)

    def run_inference(self, batch):
        input_ids = torch.stack([ex["input_ids"] for ex in batch]).to(f'cuda:{self.rank}')
        attention_mask = torch.stack([ex["attention_mask"] for ex in batch]).to(f'cuda:{self.rank}').bool()

        with torch.inference_mode():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids, use_cache=False)
        del input_ids, attention_mask
        loss = outputs.loss
        return loss

    def set_profiling_dims(self):
        self.profiling_dims = [(self.max_length - 1) // 4,
                               (self.max_length - 1) // 2,
                                self.max_length - 1]
        
    def tp_patch(self):

        # embedding
        if self.rank == 0:
            self.model.model.embed_tokens = patch_embedding(embed_tokens=self.model.model.embed_tokens,
                                                            vocab_size=self.model.config.vocab_size,
                                                            hidden_size=self.model.config.hidden_size,
                                                            padding_idx=self.tokenizer.pad_token_id,
                                                            world_size=self.world_size)

            # rmsnorm
            self.model.model.norm = patch_rmsnorm(rmsnorm=self.model.model.norm,
                                                eps=self.model.config.rms_norm_eps,
                                                world_size=self.world_size)
        
        # # ignore rope

        # Decoder Layer
        for i, layer in enumerate(self.model.model.layers):
            if self.rank == 0:
                # input_layer norm
                layer.input_layernorm = patch_rmsnorm(rmsnorm=layer.input_layernorm,
                                                    eps=self.model.config.rms_norm_eps,
                                                    world_size=self.world_size)

                # # post_attention_layernorm
                layer.post_attention_layernorm = patch_rmsnorm(rmsnorm=layer.post_attention_layernorm,
                                                            eps=self.model.config.rms_norm_eps,
                                                            world_size=self.world_size)

            # self_attn
            layer.self_attn.q_proj = patch_linear(linear=layer.self_attn.q_proj,
                                                  bias=self.model.config.attention_bias,
                                                  world_size=self.world_size,
                                                  rank=self.rank)
            layer.self_attn.k_proj = patch_linear(linear=layer.self_attn.k_proj,
                                                  bias=self.model.config.attention_bias,
                                                  world_size=self.world_size,
                                                  rank=self.rank)
            layer.self_attn.v_proj = patch_linear(linear=layer.self_attn.v_proj,
                                                  bias=self.model.config.attention_bias,
                                                  world_size=self.world_size,
                                                  rank=self.rank)
            layer.self_attn.o_proj = patch_output_linear(linear=layer.self_attn.o_proj,
                                                  bias=self.model.config.attention_bias,
                                                  world_size=self.world_size,
                                                  rank=self.rank)
            # mlp
            layer.mlp.gate_proj = patch_linear(linear=layer.mlp.gate_proj,
                                               bias=self.model.config.mlp_bias,
                                               world_size=self.world_size,
                                               rank=self.rank)
            layer.mlp.up_proj = patch_linear(linear=layer.mlp.up_proj,
                                             bias=self.model.config.mlp_bias,
                                             world_size=self.world_size,
                                             rank=self.rank)
            layer.mlp.down_proj = patch_linear(linear=layer.mlp.down_proj,
                                               bias=self.model.config.mlp_bias,
                                               world_size=self.world_size,
                                               rank=self.rank)

            
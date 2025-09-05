
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import snapshot_download

from src.inference_classes.inference_class import InferenceModel
from src.custom_nonlinear.custom_eager import LlamaEager
from src.custom_nonlinear.custom_forward import llama_forward

import torch
import types
import os
import deepspeed

class LlamaModel(InferenceModel):
    def __init__(self, model_dict, nonlinear_dict, parameter_dict, device):
        super().__init__(model_dict, nonlinear_dict, parameter_dict, device)

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

    def compute_metric(self):
        return self.compute_perplexity()
    
    def load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch.float16, attn_implementation='eager', device_map='auto', use_cache=False)
        self.max_length = self.model.config.max_position_embeddings

        if self.max_length > 4096:
            self.max_length = 4096

        # init deepspeed
        # rank = int(os.environ.get("SLURM_PROCID", 0))
        # world_size = int(os.environ.get("SLURM_NTASKS", 1))
        local_rank = int(os.environ.get("SLURM_LOCALID", 0))

        torch.cuda.set_device(local_rank)

        mp_size = torch.cuda.device_count()
        if mp_size == 0:
            raise ValueError("No GPUs available for DeepSpeed inference.")

        self.ds_model = deepspeed.init_inference(
            self.model,
            tensor_parallel={"tp_size": 2},
            dtype=torch.float16,
            replace_method='nothing',
            replace_with_kernel_inject=False,
        )

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
        input_ids = torch.stack([ex["input_ids"] for ex in batch]).to(self.device)
        attention_mask = torch.stack([ex["attention_mask"] for ex in batch]).to(self.device).bool()
        with torch.inference_mode():
            outputs = self.ds_model(input_ids=input_ids)#, attention_mask=attention_mask, labels=input_ids, use_cache=False)
        del input_ids, attention_mask
        loss = outputs.loss
        return loss

    def set_profiling_dims(self):
        self.profiling_dims = [(self.max_length - 1) // 4,
                               (self.max_length - 1) // 2,
                                self.max_length - 1]
        
    
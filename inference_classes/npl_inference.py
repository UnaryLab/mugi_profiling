import torch
import math
import gc
import shutil
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import snapshot_download
import os

from inference_classes.inference_class import InferenceModel

class NLPModel(InferenceModel):
    def __init__(self, model_dict, nonlinear_dict, parameter_dict, device):
        super().__init__(model_dict, nonlinear_dict, parameter_dict, device)

    def load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if '405' in self.model_name:
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name, offload_folder=None, offload_state_dict=True, torch_dtype=torch.float16, attn_implementation='eager', device_map='auto', use_cache=False)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch.float16, attn_implementation='eager', device_map='auto', use_cache=False)
        self.max_length = self.model.config.max_position_embeddings
        if self.max_length > 4096:
            self.max_length = 4096

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

    def set_profiling_dims(self):
        self.profiling_dims = [(self.max_length - 1) // 4,
                               (self.max_length - 1) // 2,
                                self.max_length - 1]
        
    def compute_metric(self):
        return math.exp(self.total_loss / self.num_batches)
    
    def compute_loss(self, batch):
        input_ids = torch.stack([ex["input_ids"] for ex in batch]).to(self.device)
        attention_mask = torch.stack([ex["attention_mask"] for ex in batch]).to(self.device).bool()
        with torch.inference_mode():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids, use_cache=False)
        del input_ids, attention_mask
        loss = outputs.loss
        return loss

    def cleanup(self):
        del self.model
        del self.tokenizer
        del self.inputs
        del self.dataset
        gc.collect()
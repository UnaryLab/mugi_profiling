import torch
import gc
import shutil
import types

from huggingface_hub import snapshot_download
from transformers import AutoImageProcessor, AutoModelForImageClassification

from src.inference_classes.inference_class import InferenceModel
from src.custom_nonlinear.custom_eager import SwinEager
from src.custom_nonlinear.custom_forward import swin_forward

class SwinModel(InferenceModel):
    def __init__(self, model_dict, nonlinear_dict, parameter_dict, device):
        super().__init__(model_dict, nonlinear_dict, parameter_dict, device)

    def load_model(self):
        self.processor = AutoImageProcessor.from_pretrained(self.model_name, use_fast=True)
        self.model = AutoModelForImageClassification.from_pretrained(self.model_name, torch_dtype=torch.float16, attn_implementation='eager',  device_map='auto')
        #self.max_length = self.model.config.max_source_positions

    def process_dataset(self):
        self.subset = list(self.dataset.take(self.n_samples))
        self.inputs = []

        for example in self.subset:
            img = example["image"] if 'image' in example else example['jpg']

            inputs = self.processor(
                images=img,
                return_tensors="pt"
            )

            inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)
            label = example["label"] if 'label' in example else example['cls']

            processed_example = {
                "pixel_values": inputs["pixel_values"],
                "labels": torch.tensor([label], dtype=torch.long),
                "label": label
            }

            self.inputs.append(processed_example)

    def patch_layers(self, attention_class, ffn_class, attention_parameters: dict = {}, ffn_parameters: dict = {}, attention_keys: list = [], ffn_keys: list = [], path: str = None):
        for i, layers in enumerate(self.model.swinv2.encoder.layers):
            for j, block in enumerate(block.blocks):
                block_idx = i * len(layers) + j
                layer_device = next(block.parameters()).device
                
                attention_object = attention_class(**attention_parameters, layer=block_idx, device=layer_device, profile_path=path, profile_dims=self.profile_dims, keys=attention_keys, profile=self.profile)
                ffn_object = ffn_class(**ffn_parameters, layer=block_idx, device=layer_device, profile_path=path, profile_dims=self.profile_dims, keys=ffn_keys, profile=self.profile)
                forward = swin_forward(attention_object)
                
                block.attention.self.forward = types.MethodType(forward, block.attention.self)
                block.intermediate.intermediate_act_fn = ffn_object

            # elif 'swinv2' in self.model_name:
        #     for i, block in enumerate(self.model.swinv2.encoder.layers):
        #         for j, layer in enumerate(block.blocks):
        #             layer_device = next(layer.parameters()).device
        #             if i == 0:
        #                 self.device = layer_device
        #             attention_object = attention_class(**attention_parameters, layer=j, blocks=i, device=layer_device, profile_path=path, profile_dims=self.profile_dims, keys=attention_keys, profile=self.profile)
        #             ffn_object = ffn_class(**ffn_parameters, layer=j, blocks=i, device=layer_device, profile_path=path, profile_dims=self.profile_dims, keys=ffn_keys, profile=self.profile)
        #             forward = swin_forward(attention_object)
                    
        #             layer.attention.self.forward = types.MethodType(forward, layer.attention.self)
        #             layer.intermediate.intermediate_act_fn = ffn_object

    def compute_metric(self):
        return self.compute_loss()
    
    def compute_loss(self, batch):
        pixel_values = torch.stack([ex["pixel_values"].squeeze(0) for ex in batch]).to(self.device).to(torch.float16)
        labels = torch.stack([ex["labels"] for ex in batch]).squeeze(-1).to(self.device)
        with torch.no_grad():
            outputs = self.model(pixel_values=pixel_values, labels=labels)

        del pixel_values, labels
        loss = outputs.loss
        return loss
    
    def cleanup(self):
        del self.model
        del self.processor
        del self.inputs
        del self.dataset
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        model_cache_path = snapshot_download(self.model_name, local_files_only=True)
        shutil.rmtree(model_cache_path, ignore_errors=True)

        gc.collect()
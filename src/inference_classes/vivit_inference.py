import torch
import io
import gc
import shutil
from huggingface_hub import snapshot_download
from transformers import AutoProcessor, AutoModelForVideoClassification
from decord import VideoReader, cpu
from PIL import Image

from src.inference_classes.inference_class import InferenceModel

class VivitModel(InferenceModel):
    def __init__(self, model_dict, nonlinear_dict, parameter_dict, device):
        super().__init__(model_dict, nonlinear_dict, parameter_dict, device)

    def load_model(self):
        self.processor = AutoProcessor.from_pretrained(self.model_name, use_fast=True)
        self.model = AutoModelForVideoClassification.from_pretrained(self.model_name, torch_dtype=torch.float16, attn_implementation='eager', device_map='auto')
        self.num_frames = self.model.config.num_frames
        self.max_length = self.model.config.image_size

    def process_dataset(self):
        self.subset = list(self.dataset.take(int(self.n_samples * 1.5)))
        self.inputs = []

        for i, example in enumerate(self.subset):
            if i >= self.n_samples:
                break
            video_reader = VideoReader(io.BytesIO(example['video']), ctx=cpu(0))
            frames_array = video_reader.get_batch(range(min(self.num_frames, len(video_reader)))).asnumpy()
            
            frames_pil = [Image.fromarray(frame) for frame in frames_array]
            inputs = self.processor(frames_pil, return_tensors='pt')

            inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)
            if inputs['pixel_values'].shape[1] != 32:
                continue
            label = example["label"]

            processed_example = {
                "pixel_values": inputs["pixel_values"],
                "labels": torch.tensor([label], dtype=torch.long),
                "label": label
            }

            self.inputs.append(processed_example)

    def compute_metric(self):
        return self.total_loss / self.num_batches

            # elif 'vivit' in self.model_name:
        #     for i, layer in enumerate(self.model.vivit.encoder.layer):
        #         layer_device = next(layer.parameters()).device
        #         if i == 0:
        #             self.device = layer_device
        #         attention_object = attention_class(**attention_parameters, layer=i, device=layer_device, profile_path=path, profile_dims=self.profile_dims, keys=attention_keys, profile=self.profile)
        #         ffn_object = ffn_class(**ffn_parameters, layer=i, device=layer_device, profile_path=path, profile_dims=self.profile_dims, keys=ffn_keys, profile=self.profile)
        #         eager_attn_fn = VivitEager(nonlinear_object=attention_object)
        #         forward = vivit_forward(eager_attn_fn)

        #         layer.attention.attention.forward = types.MethodType(forward, layer.attention.attention)
        #         layer.intermediate.intermediate_act_fn = ffn_object

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
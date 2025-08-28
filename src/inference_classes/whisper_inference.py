import torch
import librosa
import numpy as np
import gc
import math
import shutil
import types

from huggingface_hub import snapshot_download
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

from src.inference_classes.inference_class import InferenceModel
from src.custom_nonlinear.custom_eager import WhisperEager
from src.custom_nonlinear.custom_forward import whisper_forward

class WhisperModel(InferenceModel):
    def __init__(self, model_dict, nonlinear_dict, parameter_dict, device):
        super().__init__(model_dict, nonlinear_dict, parameter_dict, device)

    def load_model(self):
        self.processor = AutoProcessor.from_pretrained(self.model_name, use_fast=True)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(self.model_name, torch_dtype=torch.float16, attn_implementation='eager', device_map='auto', use_cache=False)
        self.target_sample_rate = self.processor.feature_extractor.sampling_rate
        self.max_source_length = self.model.config.max_source_positions
        self.max_target_length = self.model.config.max_target_positions

    def process_audio(self, audio_sample):
        audio_array = audio_sample['array']
        original_sr = audio_sample["sampling_rate"]
                                   
        # Resample to 16kHz if needed
        if original_sr != self.target_sample_rate:
            audio_array = librosa.resample(
                audio_array, 
                orig_sr=original_sr, 
                target_sr=self.target_sample_rate
            )
        
        # Pad or truncate audio to a consistent length
        # 30 seconds at 16kHz = 480,000 samples (common for Whisper)
        target_length = 30 * self.target_sample_rate  # 30 seconds
        
        if len(audio_array) > target_length:
            # Truncate if too long
            audio_array = audio_array[:target_length]
        elif len(audio_array) < target_length:
            # Pad with zeros if too short
            padding = target_length - len(audio_array)
            audio_array = np.pad(audio_array, (0, padding), mode='constant', constant_values=0)
        
        return audio_array

    def patch_layers(self, attention_class, ffn_class, attention_parameters: dict = {}, ffn_parameters: dict = {}, attention_keys: list = [], ffn_keys: list = [], path: str = None):

        for i, layer in enumerate(self.model.model.encoder.layers + self.model.model.decoder.layers):
            layer_device = next(layer.parameters()).device
            
            self.append_nonlinear_list(attention_class=attention_class,
                                       ffn_class=ffn_class,
                                       attention_parameters=attention_parameters,
                                       ffn_parameters=ffn_parameters,
                                       layer=i,
                                       device=layer_device,
                                       path=path,
                                       attention_keys=attention_keys,
                                       ffn_keys=ffn_keys)

            eager_attn_fn = WhisperEager(nonlinear_object=self.attention_objects[i])
            forward = whisper_forward(eager_attn_fn)

            layer.self_attn.forward = types.MethodType(forward, layer.self_attn)
            layer.activation_fn = self.ffn_objects[i]

    def process_dataset(self):
        print("Subset")
        subset = []
        for i, sample in enumerate(self.dataset):
            print(f"Processing example {i}")
            subset.append(sample)
            if i + 1 == self.n_samples:
                break
        print(f"Length of subset: {len(subset)}")
        self.inputs = []
        print("Processing examples...")
        for example in subset:
            print(f"Processing example: {example['audio']['path']}")
            audio_array = self.process_audio(example['audio'])

            inputs = self.processor(audio_array,
                                    sampling_rate=self.target_sample_rate,
                                    return_tensors="pt")
            
            inputs['input_features'] = inputs['input_features'].to(torch.float16)
            text = example['text']

            start_token = self.processor.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
            lang_token = self.processor.tokenizer.convert_tokens_to_ids("<|en|>")  # English
            task_token = self.processor.tokenizer.convert_tokens_to_ids("<|transcribe|>")
            notimestamps_token = self.processor.tokenizer.convert_tokens_to_ids("<|notimestamps|>")

            text_only_tokens = self.processor.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_target_length,
                add_special_tokens=False
            )['input_ids'].squeeze(0)

            full_sequence = torch.cat([
                torch.tensor([start_token, lang_token, task_token, notimestamps_token]),
                text_only_tokens,
                torch.tensor([self.processor.tokenizer.eos_token_id])
            ]).unsqueeze(0)

            text_inputs = {"input_ids": full_sequence}

            processed_example = {
                "input_features": inputs["input_features"],
                "labels": text_inputs["input_ids"],
                "text": text
            }

            self.inputs.append(processed_example)

    def compute_metric(self):
        return math.exp(self.total_loss / self.num_batches)
    
    def process_batch(self, batch):
        input_features = [i['input_features'] for i in batch]
        batch_features = torch.cat(input_features, dim=0).to(torch.float16)
        labels = [l['labels'].squeeze(0) for l in batch]
        batch_labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=self.model.config.pad_token_id)

        text = [l['text'] for l in batch]

        batch = {
            "input_features": batch_features,
            "labels": batch_labels,
            "text": text
        }

        return batch

    def compute_loss(self, batch):
        input_features = batch["input_features"].to(self.device)
        labels = batch["labels"].to(self.device)
        labels[labels == self.model.config.pad_token_id] = -100
        with torch.no_grad():
            outputs = self.model(input_features=input_features, labels=labels, use_cache=False)

        del input_features, labels
        loss = outputs.loss
        return loss

    def set_profiling_dims(self):
        self.source_profiling_dims = [(self.max_source_length - 1) // 4,
                                      (self.max_source_length - 1) // 2,
                                      (self.max_source_length - 1)]
        self.target_profiling_dims = [(self.max_target_length - 1) // 4,
                                      (self.max_target_length - 1) // 2,
                                      (self.max_target_length - 1)]
        
    def cleanup(self):
        del self.model
        del self.processor
        del self.inputs
        del self.dataset
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        model_cache_path = snapshot_download(self.model_name, local_files_only=True)
        print(model_cache_path)
        model_cache_path = model_cache_path.split('/snapshots')[0]
        shutil.rmtree(model_cache_path, ignore_errors=True)
        
        gc.collect()
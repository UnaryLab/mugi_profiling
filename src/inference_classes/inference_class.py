from datasets import load_dataset
from itertools import product
import torch
import gc
import os
import math
import pandas as pd
from tqdm import tqdm
import deepspeed
from abc import ABC, abstractmethod

from src.custom_nonlinear.custom_approx import CustomSoftmax, CustomSilu, CustomGelu, CustomFastGelu
from src.custom_nonlinear.custom_nonlinear_functions.pwl.pwl_gelu_approx import PWLGelu
from src.custom_nonlinear.custom_nonlinear_functions.pwl.pwl_mobilenet_approx import PWLMobilenet
from src.custom_nonlinear.custom_nonlinear_functions.pwl.pwl_silu_approx import PWLSilu
from src.custom_nonlinear.custom_nonlinear_functions.pwl.pwl_softmax_approx import PWLSoftmax
from src.custom_nonlinear.custom_nonlinear_functions.taylor.taylor_softmax_approx import TaylorSoftmax
from src.custom_nonlinear.custom_nonlinear_functions.vlp.vlp_gelu_approx import VLPGelu
from src.custom_nonlinear.custom_nonlinear_functions.vlp.vlp_silu_approx import VLPSilu
from src.custom_nonlinear.custom_nonlinear_functions.vlp.vlp_softmax_approx import VLPSoftmax

class InferenceModel(ABC):
    def __init__(self, model_dict, nonlinear_dict, parameter_dict, device):
        # Set device
        self.device = device
        
        # Initial dicts
        self.model_dict = model_dict
        self.nonlinear_dict = nonlinear_dict
        self.parameter_dict = parameter_dict

        # Dict configs
        self.dataset_parameters = model_dict.get('dataset')
        self.model_parameters = model_dict.get('model')
        self.inference_parameters = model_dict.get('parameters')
        self.nonlinear_parameters = model_dict.get('nonlinear')
        self.nonlinear_functions = nonlinear_dict.get('functions')
        self.nonlinear_function_parameters = nonlinear_dict.get('params')

        # Dict Items
        self.dataset_name = self.dataset_parameters.get('name')
        self.hf_path = self.dataset_parameters.get('hf_path')
        self.dataset_split = self.dataset_parameters.get('split')
        self.dataset_config = self.dataset_parameters.get('config')

        self.model_name = self.model_parameters.get('name')

        self.attn_op = self.nonlinear_parameters.get('attention')
        self.ffn_op = self.nonlinear_parameters.get('ffn')

        self.n_samples = parameter_dict.get('n_samples', 1)
        self.profile = parameter_dict.get('profile', False)
        self.batch_size = self.inference_parameters.get('batch_size', 1)
        
        # Initialize DataFrame for collecting results
        self.df = None

        self.attention_objects = []
        self.ffn_objects = []

    def init_deepspeed(self):
        n_gpus = torch.cuda.device_count()
        if n_gpus == 0:
            raise ValueError("No GPUs available for DeepSpeed inference.")

        self.ds_model = deepspeed.init_inference(
            self.model,
            mp_size=n_gpus,
            dtype=torch.float16,
            replace_with_kernel_inject=False
        )

    def append_nonlinear_list(self, attention_class, ffn_class, attention_parameters, ffn_parameters, layer, device, path, attention_keys, ffn_keys):

        if len(self.attention_objects) <= layer:
            print(layer)
            attention_object = attention_class(**attention_parameters,
                                               layer=layer,
                                               devlayerce=device,
                                               profile_path=path,
                                               profile_dims=self.profiling_dims,
                                               keys=attention_keys,
                                               profile=self.profile)
            self.attention_objects.append(attention_object)
        else:
            attention_object = self.attention_objects[layer]
            attention_object.set_params(**attention_parameters)

        if len(self.ffn_objects) <= layer:
            ffn_object = ffn_class(**ffn_parameters,
                                   layer=layer,
                                   device=device,
                                   profile_path=path,
                                   profile_dims=self.profiling_dims,
                                   keys=ffn_keys,
                                   profile=self.profile)
            self.ffn_objects.append(ffn_object)
        else:
            ffn_object = self.ffn_objects[layer]
            ffn_object.set_params(**ffn_parameters)

    def load_streaming_dataset(self):
        if self.dataset_config:
            self.dataset = load_dataset(self.hf_path, self.dataset_config, split=self.dataset_split, streaming=True, trust_remote_code=True)
        else:
            self.dataset = load_dataset(self.hf_path, split=self.dataset_split, streaming=True, trust_remote_code=True)

    def process_batch(self, batch):
        return batch

    def batch_dataset(self):
        batched_data = []
        for i in range(0, self.n_samples, self.batch_size):
            if i + self.batch_size > self.n_samples:
                batch = self.inputs[i:]
            else:
                batch = self.inputs[i:i + self.batch_size]
            batch = self.process_batch(batch)
            batched_data.append(batch)
        self.inputs = batched_data
    
    def flatten_dict(self, d: dict) -> dict:
        flat_dict = {}
        for key, value in d.items():
            if isinstance(value, list) and len(value) == 1:
                flat_dict[key] = value[0]
            elif isinstance(value, (str, int, float, bool)) or value is None:
                flat_dict[key] = value
            else:
                raise ValueError(f'Value for key "{key}" is not a single-element list.')
        return flat_dict

    def dict_value_to_list(self, d: dict) -> dict:
        if d is None:
            return {}
        list_dict = {}
        for key, value in d.items():
            if not isinstance(value, list):
                list_dict[key] = [value]
            else:
                list_dict[key] = value
        return list_dict

    def parameter_combinations(self, d: dict) -> dict:
        (keys, values) = zip(*d.items())
        combination = list(product(*values))
        result = [dict(zip(keys, combo)) for combo in combination]
        return result

    def nonlinear_combinations(self, d: dict)-> dict:
        d = {k: v + [None] for k, v in d.items()}
        (keys, values) = zip(*d.items())
        combination = list(product(*values))
        result = [dict(zip(keys, combo)) for combo in combination]
        result = [r for r in result if not all(v is None for v in r.values())]
        return result

    def compute_loss(self):
        return self.total_loss / self.num_batches
    
    def compute_perplexity(self):
        return math.exp(self.total_loss / self.num_batches)

    def run_batched_inference(self):
        self.total_loss = 0.0
        self.num_batches = 0
        for batch in self.inputs:
            batched_loss = self.run_inference(
                batch=batch
            )

            self.total_loss += batched_loss.item()
            self.num_batches += 1
            del batch, batched_loss
            torch.cuda.empty_cache()

        self.metric = self.compute_metric()

    def set_profiling_dims(self):
        self.profile_dims = -1

    def memory_profiled_forward(self, forward_fn, mem_stats_dict, name):
        def wrapper(*args, **kwargs):
            torch.cuda.reset_peak_memory_stats()
            result = forward_fn(*args, **kwargs)
            peak = torch.cuda.max_memory_allocated() / (2**20)  # MB
            mem_stats_dict[name] = peak
            return result
        return wrapper

    @abstractmethod
    def patch_layers(self):
        pass

    def patch_model(self, function_name, attention_parameters={}, ffn_parameters={}, patch_attention=True, patch_ffn=True):

        if function_name == 'torch':
            self.profile = True
        else:
            self.profile = False

        attention_keys = []
        if attention_parameters:
            for key, item in attention_parameters.items():
                attention_keys.append(key + '_' + str(item))

        ffn_keys = []
        if ffn_parameters:
            for key, item in ffn_parameters.items():
                ffn_keys.append(key + '_' + str(item))

        torch.cuda.empty_cache()
        gc.collect()

        attention_default_classes = [CustomSoftmax]
        ffn_default_classes = [CustomSilu, CustomGelu, CustomFastGelu]

        attention_class = CustomSoftmax
        if self.ffn_op == 'silu':
            ffn_class = CustomSilu
        elif self.ffn_op == 'gelu':
            ffn_class = CustomGelu
        elif self.ffn_op == 'fast_gelu':
            ffn_class = CustomFastGelu

        if function_name == 'vlp':
            if patch_attention: attention_class = VLPSoftmax

            if self.ffn_op == 'silu' and patch_ffn: ffn_class = VLPSilu
            elif (self.ffn_op == 'gelu' or self.ffn_op == 'fast_gelu') and patch_ffn: ffn_class = VLPGelu

        elif function_name == 'pwl':
            if patch_attention: attention_class = PWLSoftmax

            if self.ffn_op == 'silu' and patch_ffn: ffn_class = PWLSilu
            elif (self.ffn_op == 'gelu' or self.ffn_op == 'fast_gelu') and patch_ffn: ffn_class = PWLGelu

        elif function_name == 'pwl_mobilenet':
            if self.ffn_op == 'silu' and patch_ffn: ffn_class = PWLMobilenet

        elif function_name == 'taylor':
            if patch_attention: attention_class = TaylorSoftmax

        if attention_class in attention_default_classes:
            attention_parameters = {}
        if ffn_class in ffn_default_classes:
            ffn_parameters = {}

        attn_path = f'{function_name}_{self.attn_op}' if patch_attention else f'torch_{self.attn_op}'
        ffn_path = f'{function_name}_{self.ffn_op}' if patch_ffn else f'torch_{self.ffn_op}'
        path = f'profile/{self.model_name}/{attn_path}_{ffn_path}/'

        if self.profile:
            os.makedirs(path, exist_ok=True)

        attention_parameters = attention_parameters if attention_parameters else {}
        ffn_parameters = ffn_parameters if ffn_parameters else {}

        self.patch_layers(
            attention_class=attention_class,
            ffn_class=ffn_class,
            attention_parameters=attention_parameters,
            ffn_parameters=ffn_parameters,
            attention_keys=attention_keys,
            ffn_keys=ffn_keys,
            path=path
        )

        # if 'llama' in self.model_name:
        #     for i, layer in enumerate(self.model.model.layers):
        #         layer_device = next(layer.parameters()).device
        #         if i == 0:
        #             self.device = layer_device

        #         attention_object = attention_class(**attention_parameters, layer=i, device=layer_device, profile_path=path, profile_dims=self.profiling_dims, keys=attention_keys, profile=self.profile)
        #         ffn_object = ffn_class(**ffn_parameters, layer=i, device=layer_device, profile_path=path, profile_dims=self.profiling_dims, keys=ffn_keys, profile=self.profile)
        #         eager_attn_fn = LlamaEager(nonlinear_object=attention_object)
        #         forward = llama_forward(eager_attn_fn)
                
        #         layer.self_attn.forward = types.MethodType(forward, layer.self_attn)
        #         layer.mlp.act_fn = ffn_object

        # elif 'whisper' in self.model_name:
        #     for i, layer in enumerate(self.model.model.encoder.layers):
        #         layer_device = next(layer.parameters()).device
        #         if i == 0:
        #             self.device = layer_device
        #         attention_object = attention_class(**attention_parameters, layer=i, device=layer_device, profile_path=path, profile_dims=self.source_profiling_dims, keys=attention_keys,  profile=self.profile)
        #         ffn_object = ffn_class(**ffn_parameters, layer=i, device=layer_device, profile_path=path, profile_dims=self.source_profiling_dims, keys=ffn_keys,  profile=self.profile)
        #         eager_attn_fn = WhisperEager(nonlinear_object=attention_object)
        #         forward = whisper_forward(eager_attn_fn)

        #         layer.self_attn.forward = types.MethodType(forward, layer.self_attn)
        #         layer.activation_fn = ffn_object

        #     for i, layer in enumerate(self.model.model.decoder.layers):
        #         layer_device = next(layer.parameters()).device
        #         if i == 0:
        #             self.device = layer_device
        #         attention_object = attention_class(**attention_parameters, layer=i, device=layer_device, profile_path=path, profile_dims=self.target_profiling_dims, keys=attention_keys, profile=self.profile)
        #         ffn_object = ffn_class(**ffn_parameters, layer=i, device=layer_device, profile_path=path, profile_dims=self.target_profiling_dims, keys=ffn_keys, profile=self.profile)
        #         eager_attn_fn = WhisperEager(nonlinear_object=attention_object)
        #         forward = whisper_forward(eager_attn_fn)

        #         layer.self_attn.forward = types.MethodType(forward, layer.self_attn)
        #         layer.activation_fn = ffn_object

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
        
        self.run_batched_inference()

        torch.cuda.empty_cache()
        gc.collect()

        new_row = {
            'model': self.model_name,
            'modality': self.model_modality,
            'value': self.metric,
            'function_name': function_name,
            'patch_attention': patch_attention,
            'patch_ffn': patch_ffn,
            'attn_fn': attention_class.__name__,
            'ffn_fn': ffn_class.__name__
        }

        # Add attention parameters with prefixed column names to avoid conflicts
        if attention_parameters:
            for key, value in attention_parameters.items():
                new_row[f'attn_{key}'] = value
        
        # Add FFN parameters with prefixed column names to avoid conflicts
        if ffn_parameters:
            for key, value in ffn_parameters.items():
                new_row[f'ffn_{key}'] = value

        new_row = pd.DataFrame([new_row])

        if self.df is None:
            self.df = new_row
        else:
            self.df = pd.concat([self.df, new_row], axis=0, ignore_index=True)

    def loop_configuration(self):
        for function_name, function_operations in tqdm(self.nonlinear_functions.items(), desc='Patching configurations'):
            if 'ffn' in function_operations:
                if self.ffn_op not in function_operations['ffn']:
                    function_operations.pop('ffn', None)
                else:
                    function_operations['ffn'] = [self.ffn_op]

            if not function_operations:
                continue

            nonlinear_combinations = self.nonlinear_combinations(function_operations) if function_name != 'torch' else [function_operations]

            for nonlinear_combination in tqdm(nonlinear_combinations, desc=f'Processing {function_name} combinations'):
                nonlinear_combination = self.flatten_dict(nonlinear_combination)

                function_parameters = self.nonlinear_function_parameters.get(function_name)
                attention_parameters = function_parameters.get('attention') if function_parameters else None
                ffn_parameters = function_parameters.get('ffn') if function_parameters else None

                attn_op = nonlinear_combination.get('attention')
                ffn_op = nonlinear_combination.get('ffn')

                patch_attention = False
                patch_ffn = False

                attention_parameters = None if not attn_op else self.dict_value_to_list(attention_parameters) if attention_parameters else None
                ffn_parameters = None if not ffn_op else self.dict_value_to_list(ffn_parameters) if ffn_parameters else None

                attention_parameters = None if not attention_parameters else self.parameter_combinations(attention_parameters)
                ffn_parameters = None if not ffn_parameters else self.parameter_combinations(ffn_parameters)

                if not attn_op and not ffn_op:
                    continue
                elif (attn_op and not ffn_op) or (attn_op and ffn_op and not ffn_parameters):
                    patch_attention = True
                    if attention_parameters:
                        for attention_combination in attention_parameters:
                            self.patch_model(function_name, attention_parameters=attention_combination, patch_attention=patch_attention, patch_ffn=patch_ffn)
                    else:
                        self.patch_model(function_name, patch_attention=patch_attention, patch_ffn=patch_ffn)

                elif (not attn_op and ffn_op) or (attn_op and ffn_op and not attention_parameters):
                    patch_ffn = True
                    if ffn_parameters:
                        for ffn_combination in ffn_parameters:
                            self.patch_model(function_name, ffn_parameters=ffn_combination, patch_attention=patch_attention, patch_ffn=patch_ffn)
                    else:
                        self.patch_model(function_name, patch_attention=patch_attention, patch_ffn=patch_ffn)

                else:
                    pass
                    # patch_attention = True
                    # patch_ffn = True
                    # for attention_combination in tqdm(attention_parameters, desc='Attention combinations'):
                    #     for ffn_combination in tqdm(ffn_parameters, desc='FFN combinations'):
                    #         self.patch_model(function_name, attention_parameters=attention_combination, ffn_parameters=ffn_combination, patch_attention=patch_attention, patch_ffn=patch_ffn)
                
                # Cleanup between nonlinear combinations
                torch.cuda.empty_cache()
                gc.collect()

        # Save the collected results to CSV
        if self.df is not None:
            csv_file = f'csv/{self.model_name}/metric.csv'
            os.makedirs(os.path.dirname(csv_file), exist_ok=True)
            self.df.to_csv(csv_file, index=False)

    def cleanup(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        gc.collect()
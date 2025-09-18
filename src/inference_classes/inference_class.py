from datasets import load_dataset
from itertools import product
import torch
import gc
import os
import math
import pandas as pd
from tqdm import tqdm
from abc import ABC, abstractmethod
from copy import deepcopy

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
        self.nonlinear_function_parameters = nonlinear_dict.get('parameters')

        # Dict Items
        self.dataset_name = self.dataset_parameters.get('name')
        self.hf_path = self.dataset_parameters.get('hf_path')
        self.dataset_split = self.dataset_parameters.get('split')
        self.dataset_config = self.dataset_parameters.get('config')

        self.nonlinear_function = nonlinear_dict.get('nonlinear_function')
        self.approx_function = nonlinear_dict.get('approx_function')

        self.model_name = self.model_parameters.get('name')

        self.attn_op = self.nonlinear_parameters.get('attention')
        self.ffn_op = self.nonlinear_parameters.get('ffn')

        self.n_samples = parameter_dict.get('n_samples', 1)
        self.profile = parameter_dict.get('profile', False)
        self.batch_size = self.inference_parameters.get('batch_size', 1)

        self.patch_per_layer = self.nonlinear_dict.get('patch_per_layer', False)
        self.patched_layer = self.nonlinear_dict.get('patched_layer', None)
        self.patched_layer_path = f'{self.patched_layer}/' if self.patched_layer is not None else ''
        
        # Initialize DataFrame for collecting results
        self.df = None

        self.attention_objects = []
        self.ffn_objects = []

    def append_nonlinear_list(self, attention_class, ffn_class, layer, device, path, profiling_dims):

        attention_object = attention_class(layer=layer,
                                            device=device,
                                            profile_path=path,
                                            profile_dims=profiling_dims,
                                            profile=self.profile)
        self.attention_objects.append(attention_object)

        ffn_object = ffn_class(layer=layer,
                                device=device,
                                profile_path=path,
                                profile_dims=profiling_dims,
                                profile=self.profile)
        self.ffn_objects.append(ffn_object)

    def set_nonlinear_params(self, attention_parameters, ffn_parameters, layer, attention_keys, ffn_keys):
        self.attention_objects[layer].set_params(**attention_parameters, keys=attention_keys)
        self.ffn_objects[layer].set_params(**ffn_parameters, keys=ffn_keys)

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

    def compute_metric(self, total_loss, num_batches):
        return
    
    def compute_perplexity(self, total_loss, num_batches):
        ppl = torch.exp(total_loss / num_batches)
        return ppl

    def run_batched_inference(self):
        total_loss = torch.tensor(0, dtype=torch.float64)
        num_batches = torch.tensor(0, dtype=torch.float64)
        for batch in tqdm(self.inputs, desc="Batched Inference"):
            batched_loss = self.run_inference(
                batch=batch
            )
            total_loss += batched_loss.item()
            num_batches += 1
            del batch, batched_loss
            torch.cuda.empty_cache()
            
        self.metric = self.compute_metric(total_loss, num_batches).item()
        
    def set_profiling_dims(self):
        self.profile_dims = -1

    @abstractmethod
    def patch_layers(self):
        pass

    def patch_model(self):
        patch_attention = False
        patch_ffn = False
        if self.nonlinear_function in ['softmax', 'both']:
            patch_attention = True
        if self.nonlinear_function in ['ffn', 'both']:
            patch_ffn = True

        attention_class = CustomSoftmax
        if self.ffn_op == 'silu':
            ffn_class = CustomSilu
        elif self.ffn_op == 'gelu':
            ffn_class = CustomGelu
        elif self.ffn_op == 'fast_gelu':
            ffn_class = CustomFastGelu

        if self.approx_function == 'vlp':
            if patch_attention: attention_class = VLPSoftmax

            if self.ffn_op == 'silu' and patch_ffn: ffn_class = VLPSilu
            elif (self.ffn_op == 'gelu' or self.ffn_op == 'fast_gelu') and patch_ffn: ffn_class = VLPGelu

        elif self.approx_function == 'pwl':
            if patch_attention: attention_class = PWLSoftmax

            if self.ffn_op == 'silu' and patch_ffn: ffn_class = PWLSilu
            elif (self.ffn_op == 'gelu' or self.ffn_op == 'fast_gelu') and patch_ffn: ffn_class = PWLGelu

        elif self.approx_function == 'pwl_mobilenet':
            if self.ffn_op == 'silu' and patch_ffn: ffn_class = PWLMobilenet

        elif self.approx_function == 'taylor':
            if patch_attention: attention_class = TaylorSoftmax


        attn_path = f'{self.approx_function}_{self.attn_op}' if patch_attention else f'torch_{self.attn_op}'
        ffn_path = f'{self.approx_function}_{self.ffn_op}' if patch_ffn else f'torch_{self.ffn_op}'
        path = f'profile/{self.model_name}/{attn_path}_{ffn_path}/'

        if self.profile:
            os.makedirs(path, exist_ok=True)

        self.set_profiling_dims()

        self.attn_function = attention_class.__name__ if patch_attention else 'torch'
        self.ffn_function = ffn_class.__name__ if patch_ffn else 'torch'

        self.patch_layers(
            attention_class=attention_class,
            ffn_class=ffn_class,
            path=path
        )

    def run_layer_configuration(self, attn_params, ffn_params):
        attn_config_path = ''
        ffn_config_path = ''
        if attn_params:
            for key, value in attn_params.items():
                if not isinstance(value, list):
                    attn_config_path += f'{key}_{value}/'
            attn_config_path += 'layer_config/'
        if ffn_params:
            for key, value in ffn_params.items():
                if not isinstance(value, list):
                    ffn_config_path += f'{key}_{value}/'
            ffn_config_path += 'layer_config/'

        # separate layer specific parameters
        attn_global_params = {}
        attn_layer_params = {}
        attn_default_params = {}
        ffn_global_params = {}
        ffn_layer_params = {}
        ffn_default_params = {}
        attn_layer_key = None
        ffn_layer_key = None
        attn_layer_value = None
        ffn_layer_value = None

        for key, value in attn_params.items():
            if not isinstance(value, dict):
                attn_global_params[key] = value
            else:
                attn_layer_key = key
                for subkey, subvalue in value.items():
                    if subkey == 'value':
                        assert self.patched_layer not in attn_layer_params, "Multiple 'value' keys found in attention parameters."
                        attn_layer_params[self.patched_layer] = {key: subvalue}
                        attn_layer_value = subvalue
                    elif subkey == 'default':
                        attn_default_params = {key: subvalue}
                    else:
                        assert subkey not in attn_layer_params, "Multiple 'value' keys found in attention parameters."
                        assert isinstance(subkey, int), "Attention layer keys must be integers."
                        attn_layer_params[subkey - 1] = {key: subvalue}

        for key, value in ffn_params.items():
            if not isinstance(value, dict):
                ffn_global_params[key] = value
            else:
                ffn_layer_key = key
                for subkey, subvalue in value.items():
                    if subkey == 'value':
                        assert self.patched_layer not in ffn_layer_params, "Multiple 'value' keys found in FFN parameters."
                        ffn_layer_params[self.patched_layer] = {key: subvalue}
                        ffn_layer_value = subvalue
                    elif subkey == 'default':
                        ffn_default_params = {key: subvalue}
                    else:
                        assert subkey not in ffn_layer_params, "Multiple 'value' keys found in FFN parameters."
                        ffn_layer_params[subkey] = {key: subvalue}

        if attn_params:
            for i, attn_layer in enumerate(self.attention_objects):
                if i in attn_layer_params:
                    attn_layer.set_params(**attn_global_params, **attn_layer_params[i], config_path=attn_config_path)
                else:
                    attn_layer.set_params(**attn_global_params, **attn_default_params, config_path=attn_config_path)

        if ffn_params:
            for i, ffn_layer in enumerate(self.ffn_objects):
                if i in ffn_layer_params:
                    ffn_layer.set_params(**ffn_global_params, **ffn_layer_params[i], config_path=ffn_config_path)
                else:
                    ffn_layer.set_params(**ffn_global_params, **ffn_default_params, config_path=ffn_config_path)

        self.run_batched_inference()

        # torch.cuda.empty_cache()
        # gc.collect()

        new_row = {
            'model': self.model_name,
            'value': self.metric,
            'function_name': self.approx_function,
            'attn_fn': self.attn_function,
            'ffn_fn': self.ffn_function,
            'config_layer': self.patched_layer,
            f'attn_layer_{attn_layer_key}': attn_layer_value,
            f'ffn_layer_{ffn_layer_key}': ffn_layer_value
        }

        # Add attention parameters with prefixed column names to avoid conflicts
        if attn_params:
            for key, value in attn_global_params.items():
                new_row[f'attn_{key}'] = value

        if attn_layer_params:
            for layer, params in attn_layer_params.items():
                print(layer, params)
        exit()

        
        # Add FFN parameters with prefixed column names to avoid conflicts
        if ffn_params:
            for key, value in ffn_global_params.items():
                new_row[f'ffn_{key}'] = value

        new_row = pd.DataFrame([new_row])

        if self.df is None:
            self.df = new_row
        else:
            self.df = pd.concat([self.df, new_row], axis=0, ignore_index=True)

    def run_configuration(self, attn_params, ffn_params):
        
        attn_config_path = ''
        ffn_config_path = ''
        if attn_params:
            for key, value in attn_params.items():
                attn_config_path += f'{key}_{value}/'
        if ffn_params:
            for key, value in ffn_params.items():
                ffn_config_path += f'{key}_{value}/'

        for attn_layer, ffn_layer in zip(self.attention_objects, self.ffn_objects):
            attn_layer.set_params(**attn_params, config_path=attn_config_path)
            ffn_layer.set_params(**ffn_params, config_path=ffn_config_path)

        self.run_batched_inference()

        # torch.cuda.empty_cache()
        # gc.collect()

        new_row = {
            'model': self.model_name,
            'value': self.metric,
            'function_name': self.approx_function,
            'attn_fn': self.attn_function,
            'ffn_fn': self.ffn_function
        }

        # Add attention parameters with prefixed column names to avoid conflicts
        if attn_params:
            for key, value in attn_params.items():
                new_row[f'attn_{key}'] = value
        
        # Add FFN parameters with prefixed column names to avoid conflicts
        if ffn_params:
            for key, value in ffn_params.items():
                new_row[f'ffn_{key}'] = value

        new_row = pd.DataFrame([new_row])

        if self.df is None:
            self.df = new_row
        else:
            self.df = pd.concat([self.df, new_row], axis=0, ignore_index=True)

    def inference_configurations(self):
        if self.nonlinear_function_parameters:
            # Manually patch per layer (1 configuration)
            if self.patch_per_layer:

                for key, value in self.nonlinear_function_parameters.items():
                    if isinstance(value, dict):
                        config_key = key
                        runs = len(value.get('value'))

                for run in tqdm(range(runs), desc="Running configurations"):
                    params = deepcopy(self.nonlinear_function_parameters)
                    params[config_key]['value'] = value.get('value')[run]

                    attn_params = params if self.nonlinear_function in ['softmax', 'both'] else {}
                    ffn_params = params if self.nonlinear_function in ['ffn', 'both'] else {}
                    self.run_layer_configuration(attn_params=attn_params, ffn_params=ffn_params)
            # Loop through all combinations (same patch for all layers)
            else:
                for key, value in self.nonlinear_function_parameters.items():
                    if not isinstance(value, list):
                        self.nonlinear_function_parameters[key] = [value]

                (keys, values) = zip(*self.nonlinear_function_parameters.items())
                combinations = list(product(*values))
                combinations = [dict(zip(keys, combo)) for combo in combinations]
                
                for combination in tqdm(combinations, desc="Running configurations"):
                    if self.nonlinear_function in ['softmax', 'both']:
                        attn_params = combination
                    else:
                        attn_params = {}
                    if self.nonlinear_function in ['ffn', 'both']:
                        ffn_params = combination
                    else:
                        ffn_params = {}
                    self.run_configuration(attn_params=attn_params, ffn_params=ffn_params)
        # Single configuration
        else:
            self.run_configuration(attn_params={}, ffn_params={})
        
    def cleanup(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        gc.collect()
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
from custom_nonlinear.custom_nonlinear_functions.vlp.vlp_silu_approx import VLPSilu
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
        
        # Initialize DataFrame for collecting results
        self.df = None

        self.attention_objects = []
        self.ffn_objects = []

    def init_deepspeed(self):

        rank = int(os.environ.get("SLURM_PROCID", 0))
        world_size = int(os.environ.get("SLURM_NTASKS", 1))
        local_rank = int(os.environ.get("SLURM_LOCALID", 0))

        torch.cuda.set_device(local_rank)

        tp_size = torch.cuda.device_count()
        if tp_size == 0:
            raise ValueError("No GPUs available for DeepSpeed inference.")

        self.ds_model = deepspeed.init_inference(
            self.model,
            dtype=torch.float16,
            replace_with_kernel_inject=False,
            tensor_parallel={"tp_size": tp_size}
        )

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

    def run_batched_inference(self):
        total_loss = torch.tensor(0, dtype=torch.float64)
        num_batches = torch.tensor(0, dtype=torch.float64)
        for batch in self.inputs:
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

    def patch_model_dep(self, function_name, attention_parameters={}, ffn_parameters={}, patch_attention=True, patch_ffn=True):

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

        # self.run_batched_inference()

        # torch.cuda.empty_cache()
        # gc.collect()

        # new_row = {
        #     'model': self.model_name,
        #     'value': self.metric,
        #     'function_name': function_name,
        #     'patch_attention': patch_attention,
        #     'patch_ffn': patch_ffn,
        #     'attn_fn': attention_class.__name__,
        #     'ffn_fn': ffn_class.__name__
        # }

        # # Add attention parameters with prefixed column names to avoid conflicts
        # if attention_parameters:
        #     for key, value in attention_parameters.items():
        #         new_row[f'attn_{key}'] = value
        
        # # Add FFN parameters with prefixed column names to avoid conflicts
        # if ffn_parameters:
        #     for key, value in ffn_parameters.items():
        #         new_row[f'ffn_{key}'] = value

        # new_row = pd.DataFrame([new_row])

        # if self.df is None:
        #     self.df = new_row
        # else:
        #     self.df = pd.concat([self.df, new_row], axis=0, ignore_index=True)

    #def run_configuration(self, function_name, attention_parameters={}, ffn_parameters={}):
        
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

        self.patch_layers(
            attention_class=attention_class,
            ffn_class=ffn_class,
            path=path
        )

    def loop_configuration(self):
        for key, value in self.nonlinear_function_parameters.items():
            if not isinstance(value, list):
                self.nonlinear_function_parameters[key] = [value]

        (keys, values) = zip(*self.nonlinear_function_parameters.items())
        combinations = list(product(*values))
        combinations = [dict(zip(keys, combo)) for combo in combinations]
        
        for combination in combinations:
            print(combination)
            exit()

        

    def cleanup(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        gc.collect()
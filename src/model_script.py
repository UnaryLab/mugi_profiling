import argparse
import yaml
import torch
import os
import pandas as pd

from src.utils import validate_config
from src.inference_classes.whisper_inference import WhisperModel
from src.inference_classes.llama_inference import LlamaModel
from src.inference_classes.vivit_inference import VivitModel
from src.inference_classes.swin_inference import SwinModel

token = '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'

def evaluate_model(model_dict, nonlinear_dict, parameter_dict, nonlinear_config_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Validate configurations
    model_name = validate_config(model_dict, nonlinear_dict, parameter_dict)
    model_name_lower = model_name.lower()

    nonlinear_config_path = nonlinear_config_path.split('/')[-1].split('.')[0]

    inference_model = None
    # Initialize model class
    if 'llama' in model_name_lower:
        inference_model = LlamaModel(model_dict, nonlinear_dict, parameter_dict, device)
    elif 'whisper' in model_name_lower:
        inference_model = WhisperModel(model_dict, nonlinear_dict, parameter_dict, device)
    elif 'swin' in model_name_lower:
        inference_model = SwinModel(model_dict, nonlinear_dict, parameter_dict, device)
    elif 'vivit' in model_name_lower:
        inference_model = VivitModel(model_dict, nonlinear_dict, parameter_dict, device)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    inference_model.csv_file = f'output/csv/{inference_model.model_name}/{nonlinear_config_path}/metric.csv'
    if os.path.exists(inference_model.csv_file):
        inference_model.df = pd.read_csv(inference_model.csv_file)
    else:
        os.makedirs(os.path.dirname(inference_model.csv_file), exist_ok=True)
        inference_model.df = None

    print(f'Running inference for model: {inference_model.model_name}')
    print('Loading model...')
    inference_model.load_model()
    print('Patching model...')
    inference_model.patch_model()
    print('Initializing deepspeed...')
    inference_model.init_deepspeed()
    print('Loading dataset...')
    inference_model.load_streaming_dataset()
    print('Processing dataset...')
    inference_model.process_dataset()
    print('Batching dataset...')
    inference_model.batch_dataset()
    inference_model.set_profiling_dims()
    print('Patching layers...')
    exit()
    inference_model.loop_configuration()

    inference_model.df.to_csv(inference_model.csv_file, index=False)

def main():
    parser = argparse.ArgumentParser(description="Run profiling on transformer model with custom nonlinear functions.")
    
    parser.add_argument('--model_config', type=str, default=None, 
                        help='Path to model config YAML file (default: None)')
    parser.add_argument('--nonlinear_config', type=str, default=None,
                        help='Path to nonlinear config YAML file (default: None)')
    parser.add_argument('--parameter_config', type=str, default=None,
                        help='Path to inference parameters YAML file (default: None)')
    parser.add_argument('--hf_token', type=str, default=None,
                        help='Hugging Face token for authentication (default: None, assumes hf is already logged in)')
    args, unknown = parser.parse_known_args()


    print("Loading configuration files...")

    model_config = yaml.safe_load(open(args.model_config))
    nonlinear_config = yaml.safe_load(open(args.nonlinear_config))
    parameter_config = yaml.safe_load(open(args.parameter_config))

    print("Successfully loaded configuration files.")

    evaluate_model(model_config, nonlinear_config, parameter_config, args.nonlinear_config)


if __name__ == '__main__':
    main()
from huggingface_hub import login

def huggingface_login(token):
    """
    Login to Hugging Face.
    """
    try:
        login(token=token)
        print("Successfully logged in to Hugging Face.")
    except:
        print("HF_TOKEN invalid or not set.")
        exit()

def validate_config(model_dict, nonlinear_dict, parameter_dict):
    dataset_parameters = model_dict.get('dataset')
    assert isinstance(dataset_parameters, dict), "Dataset configuration is not defined or does not contain dataset parameters."

    model_parameters = model_dict.get('model')
    assert isinstance(model_dict, dict), "Model configuration is not defined or does not contain model."

    nonlinear_parameters = model_dict.get('nonlinear')
    assert isinstance(nonlinear_parameters, dict), "Nonlinear operations are not defined or do not contain operations."

    nonlinear_function = nonlinear_dict.get('function')
    nonlinear_function_parameters = nonlinear_dict.get('params')

    dataset_name = dataset_parameters.get('name')
    hf_path = dataset_parameters.get('hf_path')

    model_name = model_parameters.get('name')

    attn_op = nonlinear_parameters.get('attention')
    ffn_op = nonlinear_parameters.get('ffn')

    assert dataset_name is not None, "Dataset name is not defined."
    assert hf_path is not None, "Hugging Face path for dataset is not defined."
    assert model_name is not None, "Model name is not defined."
    assert attn_op is not None, "Attention operation is not defined."
    assert ffn_op is not None, "Feed-Forward network operation is not defined."

    print('\t|', model_name, '|')

    return model_name
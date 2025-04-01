import torch

def get_gpt2_config(model_size='small'):
    configs = {
        'small': {  # GPT-2 Small (124M)
            'vocab_size': 50257,
            'context_length': 1024,
            'emb_dim': 768,
            'n_layers': 12,
            'n_heads': 12,
            'drop_rate': 0.1,
            'qkv_bias': True
        },
        'medium': {  # GPT-2 Medium (355M)
            'vocab_size': 50257,
            'context_length': 1024,
            'emb_dim': 1024,
            'n_layers': 24,
            'n_heads': 16,
            'drop_rate': 0.1,
            'qkv_bias': True
        },
        'large': {  # GPT-2 Large (774M)
            'vocab_size': 50257,
            'context_length': 1024,
            'emb_dim': 1280,
            'n_layers': 36,
            'n_heads': 20,
            'drop_rate': 0.1,
            'qkv_bias': True
        },
        'xl': {  # GPT-2 XL (1.5B)
            'vocab_size': 50257,
            'context_length': 1024,
            'emb_dim': 1600,
            'n_layers': 48,
            'n_heads': 25,
            'drop_rate': 0.1,
            'qkv_bias': True
        }
    }
    
    if model_size not in configs:
        raise ValueError(f"Model size {model_size} not supported")
    
    return configs[model_size]

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
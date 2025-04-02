import torch
import json

class GPTConfig:
    def __init__(self, 
                 vocab_size= 50257,   # default values
                 context_length= 1024, 
                 emb_dim= 768, 
                 n_layers= 12, 
                 n_heads=12, 
                 drop_rate=0.1, 
                 qkv_bias=True,
                 model_size='small'  # default model size
                 ):
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.emb_dim = emb_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.drop_rate = drop_rate
        self.qkv_bias = qkv_bias
        self.model_size = model_size

        assert emb_dim%n_heads == 0, "Embedding dimension must be divisible by number of heads"
        self.head_dim = emb_dim // n_heads

        # expansion factor for feed forward network
        self.ffn_dim = 4*emb_dim

    @classmethod
    def from_pretrained(cls, weights_path):
        weights = torch.load(weights_path, map_location='cpu')
        if 'config' in weights:
            config_dict = weights['config']
            return cls(
                vocab_size=config_dict.get("vocab_size", 50257),
                context_length=config_dict.get("context_length", 1024),
                emb_dim=config_dict.get("emb_dim", 768),
                n_layers=config_dict.get("n_layers", 12),
                n_heads=config_dict.get("n_heads", 12),
                drop_rate=config_dict.get("drop_rate", 0.1),
                qkv_bias=config_dict.get("qkv_bias", True),
                model_type='custom',
            )
        else:
            raise ValueError("No config found in the provided weights path.")
        
    @classmethod
    def from_model_size(cls, model_size='small'):
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
        }
        if model_size not in configs:
            raise ValueError(f"Model size {model_size} not supported")
        config_dict = configs[model_size]
        config_dict['model_type'] = model_size
        return cls(**config_dict)
    
    def to_dict(self):
        return {
            'vocab_size': self.vocab_size,
            'context_length': self.context_length,
            'emb_dim': self.emb_dim,
            'n_layers': self.n_layers,
            'n_heads': self.n_heads,
            'drop_rate': self.drop_rate,
            'qkv_bias': self.qkv_bias,
            'model_type': self.model_size,
            'head_dim': self.head_dim,
            'ffn_dim': self.ffn_dim,
            'model_type': self.model_type
        }
    
    def save(self, config_path):
        config_dict = self.to_dict()
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=4)

    @classmethod
    def load(cls, config_path):
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)
    
    def __repr__(self):
        return f'GPTConfig(type={self.model_type}, layers={self.n_layers}, emb_dim={self.emb_dim}, heads={self.n_heads})'
    

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
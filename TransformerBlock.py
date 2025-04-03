from MHA import CausalMultiHeadAttention
import torch
import torch.nn as nn
from config import GPT2Config

# MLP or feed forward network with gelu activations
class MLP(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()

        self.fc1 = nn.Linear(config.emb_dim, 4*config.emb_dim)
        # approx gelu to match with original hf gpt2
        self.gelu = nn.GELU(approximate='tanh')
        self.fc2 = nn.Linear(4*config.emb_dim, config.emb_dim)
        self.dropout = nn.Dropout(config.dropout_res)

    def forward(self, x):
        return self.dropout(self.fc2(self.gelu(self.fc1(x))))


#Transformer Block
class Block(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.emb_dim)
        self.attn = CausalMultiHeadAttention(config)
        self.ln2= nn.LayerNorm(config.emb_dim)
        self.mlp = MLP(config)

    def forward(self, x):
        # attn block
        # pre-layer norm (opposite to original transformer model)
        x = x + self.attn(self.ln1(x))
        #mlp block
        x = x + self.mlp(self.ln2(x))
        return x
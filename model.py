import torch
import torch.nn as nn
from Transformer import TransformerBlock, LayerNorm

class GPTModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        self.tok_emb = nn.Embedding(config.vocab_size, config.emb_dim)
        self.pos_emb = nn.Embedding(config.context_length, config.emb_dim)
        self.drop_emb = nn.Dropout(config.drop_rate)

        self.trf_blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])
        
        self.final_norm = LayerNorm(config.emb_dim)
        self.out_head = nn.Linear(config.emb_dim, config.vocab_size, bias=False)

    
    def forward(self, input_ids):
        batch_size, seq_len = input_ids.shape
        tok_embs = self.tok_emb(input_ids)

        pos_ids = torch.arange(seq_len, device=input_ids.device)
        pos_embs = self.pos_emb(pos_ids)

        x = tok_embs + pos_embs
        x = self.drop_emb(x)

        for block in self.trf_blocks:
            x = block(x)

        x = self.final_norm(x)
        logits = self.out_head(x)
        
        return logits
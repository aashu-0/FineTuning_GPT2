import torch
import torch.nn as nn
import math
from config import GPT2Config
import torch.nn.functional as F


# for attention -> using flash attention implementation by pytorch
class CausalMultiHeadAttention(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.n_heads = config.n_heads
        self.emb_dim = config.emb_dim
        self.head_dim = self.emb_dim//self.n_heads

        assert self.emb_dim % self.n_heads == 0

        self.qkv = nn.Linear(self.emb_dim, 3*self.emb_dim)
        self.out_proj = nn.Linear(self.emb_dim, self.emb_dim)  # layer to combine head outputs
        self.attn_dropout_p = config.dropout_attn
        self.res_dropout = nn.Dropout(config.dropout_res)
        # self.register_buffer(
        #     'mask', torch.triu(torch.ones(config.context_length, config.context_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, c = x.size()

        qkv = self.qkv(x)
        q,k,v = qkv.split(self.emb_dim, dim=2) # splits along 3rd dim

        # .view() -> to reshape tensors
        # split d_out into n_heads, head_dim
        # (b, num_tokens, c) -> (b, num_tokens, n_heads, head_dim)
        # permute to convert (b, num_tokens, n_heads, head_dim) -> (b, n_heads, num_tokens, head_dim)

        k = k.view(b, num_tokens, self.n_heads, self.head_dim).permute(0,2,1,3)
        q = q.view(b, num_tokens, self.n_heads, self.head_dim).permute(0,2,1,3)
        v = v.view(b, num_tokens, self.n_heads, self.head_dim).permute(0,2,1,3)

        # attn_scores = q @ k.transpose(2,3)
        # mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        # attn_scores.masked_fill_(mask_bool, -torch.inf)
        # attn_weights = torch.softmax(attn_scores / math.sqrt(k.shape[-1]), dim = -1)
        # attn_weights = self.dropout(attn_weights)

        ## efficient flash attention
        weights= F.scaled_dot_product_attention(
            q,k,v,
            is_causal=True, # for casual masking
            dropout_p= self.attn_dropout_p if self.training else 0.0 #apply dropout to attn_wei only during training.
        )
        output = weights.transpose(1,2)
        # transposing makes tensor non-contiguous

        # therefore before flattening into shape (b, num_tokens, self.emb_dim) make into contiguous 
        output = output.contiguous().view(b, num_tokens, self.emb_dim)  # self.emb_dim = self.n_heads * self.head_dim

        output = self.out_proj(output)
        # residual dropout
        output = self.res_dropout(output)
        return output
class GPT2Config:
    def __init__(
        self,
        vocab_size=50257,  
        context_length=1024,
        emb_dim=768,
        n_heads=12,
        n_layers=12,
        dropout_emb=0.1,
        dropout_attn = 0.1,
        dropout_res=0.1,
        layernorm_eps=1e-5,
        qkv_bias=True,
    ):
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.emb_dim = emb_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dropout_emb = self.dropout_emb
        self.dropout_res = self.dropout_res
        self.dropout_attn = self.dropout_attn
        self.layernorm_eps = self.layernorm_eps
        self.qkv_bias = qkv_bias
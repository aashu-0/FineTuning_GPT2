import torch
import torch.nn as nn
from base_model.TransformerBlock import Block
from base_model.config import GPT2Config

class GPTModel(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.config = config

        self.tok_emb = nn.Embedding(config.vocab_size, config.emb_dim)

        self.pos_emb = nn.Embedding(config.context_length, config.emb_dim)

        self.emb_dropout= nn.Dropout(config.dropout_emb)


        self.trf_blocks = nn.ModuleList([
            Block(config) for _ in range(config.n_layers)
        ])
        
        self.final_norm = nn.LayerNorm(config.emb_dim)
        # for gpt2 no bias
        self.out_head = nn.Linear(config.emb_dim, config.vocab_size, bias=False)

    
    def forward(self, input_ids):
        _, current_seq_len = input_ids.size()
        self.pos_ids = torch.arange(config.context_length, device= input_ids.device)
        positions = self.pos_ids[:,:current_seq_len]

        x = self.tok_emb(input_ids) + self.pos_emb(positions)
        x = self.emb_dropout(x)

        for block in self.trf_blocks:
            x = block(x)

        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
    

if __name__ == "__main__":
    from base_model.utils import text_to_token_ids, token_ids_to_text, generate
    import torch
    import tiktoken

    tokenizer = tiktoken.get_encoding('gpt2')
    config = GPT2Config()
    gpt2 = GPTModel(config)
    input_ids = text_to_token_ids('Hello too much buttering, sama', tokenizer)

    out = gpt2(input_ids)
    # print(out)
    #print(out.size())
    out_ids = generate(model=gpt2,
                       idx=input_ids,
                       max_new_tokens=35,
                       context_size= config.context_length,
                       eos_id= 50256,
                       temp=0.7,
                       top_k=4)
    text = token_ids_to_text(out_ids, tokenizer)
    print(text)

    # Output got:
    # Hello too much buttering, samaitemsQual Brilliant narfooted Ask thirstyzers alluded bytesrical Nish trucks MA amounts confess712akespequer Recall separatist
    # conventionashionclientpleted uncomcaliber Δin subsequent coaxammefully advisorsmist
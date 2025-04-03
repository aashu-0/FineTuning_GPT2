import torch
from transformers import GPT2LMHeadModel
from model import GPTModel
import numpy as np
from config import GPT2Config

def load_gpt2_weights_to_model(config: GPT2Config):

    # loading pre-trained GPT-2 model
    print("Loading pre-trained GPT-2 model from Hugging Face...")
    model_hf = GPT2LMHeadModel.from_pretrained('gpt2')

    config = GPT2Config()
    # initialize
    print("Initializing custom model...")
    model = GPTModel(config)
    
    # --------MAPPING---------
    with torch.no_grad():
        model.tok_emb.weight.copy_(model_hf.transformer.wte.weight)
        model.pos_emb.weight.copy_(model_hf.transformer.wpe.weight)

        for block_idx in range(len(model.trf_blocks)):
            model.trf_blocks[block_idx].ln1.weight.copy_(
                model_hf.transformer.h[block_idx].ln_1.weight
            )
            model.trf_blocks[block_idx].ln1.bias.copy_(
                model_hf.transformer.h[block_idx].ln_1.bias
            )
            # HF uses conv1d for qkv proj, we use linear => transpose

            model.trf_blocks[block_idx].attn.qkv.weight.copy_(
                model_hf.transformer.h[block_idx].attn.c_attn.weight.t()
            )
            model.trf_blocks[block_idx].attn.qkv.bias.copy_(
                model_hf.transformer.h[block_idx].attn.c_attn.bias
            )
            model.trf_blocks[block_idx].attn.out_proj.weight.copy_(
                model_hf.transformer.h[block_idx].attn.c_proj.weight.t()
            )
            model.trf_blocks[block_idx].attn.out_proj.bias.copy_(
                model_hf.transformer.h[block_idx].attn.c_proj.bias
            )
            model.trf_blocks[block_idx].ln2.weight.copy_(
                model_hf.transformer.h[block_idx].ln_2.weight
            )
            model.trf_blocks[block_idx].ln2.bias.copy_(
                model_hf.transformer.h[block_idx].ln_2.bias
            )
            model.trf_blocks[block_idx].mlp.fc1.weight.copy_(
                model_hf.transformer.h[block_idx].mlp.c_fc.weight.t()
            )
            model.trf_blocks[block_idx].mlp.fc1.bias.copy_(
                model_hf.transformer.h[block_idx].mlp.c_fc.bias
            )
            model.trf_blocks[block_idx].mlp.fc2.weight.copy_(
                model_hf.transformer.h[block_idx].mlp.c_proj.weight.t()
            )
            model.trf_blocks[block_idx].mlp.fc2.bias.copy_(
                model_hf.transformer.h[block_idx].mlp.c_proj.bias
            )
        model.final_norm.weight.copy_(model_hf.transformer.ln_f.weight)
        model.final_norm.bias.copy_(model_hf.transformer.ln_f.bias)

        model.out_head.weight.copy_(model_hf.lm_head.weight)
        # no bias in out head in gpt2

        print("Weights successfully loaded into custom model.")
        return model

# example
if __name__ == "__main__":

    from config import GPT2Config
    from utils import text_to_token_ids, token_ids_to_text, generate
    import tiktoken
    import torch
    
    torch.manual_seed(123)
    config = GPT2Config()
    model = load_gpt2_weights_to_model(config)
    
    tokenizer = tiktoken.get_encoding('gpt2')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print(device)
    # test pass
    # test_input = torch.randint(0, config.vocab_size, (1, 10))
    # with torch.no_grad():
    #     output = model(test_input)
    # print(f"Output shape: {output.shape}")  # should be [1, 10, 50257]

    # generate some text from the model given a start context
    input_text = 'Hello, GPT2'

    token_ids = generate(
        model = model,
        idx = text_to_token_ids(input_text, tokenizer).to(device),
        max_new_tokens =35,
        context_size= config.context_length,
        eos_id= 50256,
        temp=0.7,
        top_k=30)
    generated_text =token_ids_to_text(token_ids, tokenizer)
    print(generated_text)

    # output yayy!!
    # Hello, GPT2 is a game with a whole lot of fun and great gameplay. If I have to stop playing it for a 
    # few minutes, don't worry, I'm going to be back

    # well gpt2 is not a game :)
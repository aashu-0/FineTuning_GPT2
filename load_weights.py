import torch
from transformers import GPT2LMHeadModel
from model import GPTModel
import numpy as np

def convert_gpt2_weights_to_custom(config):
    """
    load weights from pre-trained GPT-2 model into custom model architecture.
    """
    # loading pre-trained GPT-2 model
    print("Loading pre-trained GPT-2 model...")
    gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')
    
    # initialize
    print("Initializing custom model...")
    custom_model = GPTModel(config)
    
    # state dicts
    gpt2_state_dict = gpt2_model.state_dict()
    custom_state_dict = custom_model.state_dict()
    
    # new state dict for custom model
    new_state_dict = {}
    
    # --------MAPPING---------

    # map token embeds
    new_state_dict['tok_emb.weight'] = gpt2_state_dict['transformer.wte.weight']
    
    # map pos embeds
    new_state_dict['pos_emb.weight'] = gpt2_state_dict['transformer.wpe.weight'][:config.context_length, :]
    
    # map transformer blocks
    for i in range(config.n_layers):
       
        # attn weights
        # split Q,K,V for our architecture
        qkv_weight = gpt2_state_dict[f'transformer.h.{i}.attn.c_attn.weight']
        qkv_bias = gpt2_state_dict[f'transformer.h.{i}.attn.c_attn.bias']
        
        # split the combined qkv

        head_dim = config.emb_dim // config.n_heads
        qkv_weight_chunks = qkv_weight.chunk(3, dim=1)
        qkv_bias_chunks = qkv_bias.chunk(3, dim=0)

        q_weight, k_weight, v_weight = qkv_weight_chunks
        q_bias, k_bias, v_bias = qkv_bias_chunks
        
        # map
        new_state_dict[f'trf_blocks.{i}.attn.W_query.weight'] = q_weight
        new_state_dict[f'trf_blocks.{i}.attn.W_key.weight'] = k_weight
        new_state_dict[f'trf_blocks.{i}.attn.W_value.weight'] = v_weight
        
        if config.qkv_bias:
            new_state_dict[f'trf_blocks.{i}.attn.W_query.bias'] = q_bias
            new_state_dict[f'trf_blocks.{i}.attn.W_key.bias'] = k_bias
            new_state_dict[f'trf_blocks.{i}.attn.W_value.bias'] = v_bias
        
        # out proj
        new_state_dict[f'trf_blocks.{i}.attn.out_proj.weight'] = gpt2_state_dict[f'transformer.h.{i}.attn.c_proj.weight']
        new_state_dict[f'trf_blocks.{i}.attn.out_proj.bias'] = gpt2_state_dict[f'transformer.h.{i}.attn.c_proj.bias']
        
        # layer norms
        new_state_dict[f'trf_blocks.{i}.norm1.scale'] = gpt2_state_dict[f'transformer.h.{i}.ln_1.weight']
        new_state_dict[f'trf_blocks.{i}.norm1.shift'] = gpt2_state_dict[f'transformer.h.{i}.ln_1.bias']
        new_state_dict[f'trf_blocks.{i}.norm2.scale'] = gpt2_state_dict[f'transformer.h.{i}.ln_2.weight']
        new_state_dict[f'trf_blocks.{i}.norm2.shift'] = gpt2_state_dict[f'transformer.h.{i}.ln_2.bias']
        
        # ffn
        new_state_dict[f'trf_blocks.{i}.ff.layers.0.weight'] = gpt2_state_dict[f'transformer.h.{i}.mlp.c_fc.weight'].t()
        new_state_dict[f'trf_blocks.{i}.ff.layers.0.bias'] = gpt2_state_dict[f'transformer.h.{i}.mlp.c_fc.bias']
        new_state_dict[f'trf_blocks.{i}.ff.layers.2.weight'] = gpt2_state_dict[f'transformer.h.{i}.mlp.c_proj.weight'].t()
        new_state_dict[f'trf_blocks.{i}.ff.layers.2.bias'] = gpt2_state_dict[f'transformer.h.{i}.mlp.c_proj.bias']
    
    # final norm
    new_state_dict['final_norm.scale'] = gpt2_state_dict['transformer.ln_f.weight']
    new_state_dict['final_norm.shift'] = gpt2_state_dict['transformer.ln_f.bias']
    
    # out
    new_state_dict['out_head.weight'] = gpt2_state_dict['lm_head.weight']
    
    # load weights into custom model
    custom_model.load_state_dict(new_state_dict, strict=False)
    
    print("Weights successfully loaded into custom model.")
    return custom_model


# example
if __name__ == "__main__":

    from config import GPTConfig
    from utils import text_to_token_ids, token_ids_to_text, generate
    import tiktoken
    import torch
    
    # config model to match gpt2 small config
    config = GPTConfig(
        vocab_size=50257,
        context_length=1024,
        emb_dim=768,
        n_heads=12,
        n_layers=12,
        drop_rate=0.1,
        qkv_bias=True
    )
    
    model = convert_gpt2_weights_to_custom(config)
    print(model)
    
    tokenizer = tiktoken.get_encoding('gpt2')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    # test
    # test_input = torch.randint(0, config.vocab_size, (1, 10))
    # with torch.no_grad():
    #     output = model(test_input)
    # print(f"Output shape: {output.shape}")  # should be [1, 10, 50257]

    # generate some text from the model given a start context
    input_text = '1234'

    token_ids = generate(
        model = model,
        idx = text_to_token_ids(input_text, tokenizer).to(device),
        max_new_tokens =35,
        context_size= config.context_length,
        eos_id= 50256,
        temp=3,
        top_k=1)
    generated_text =token_ids_to_text(token_ids, tokenizer)
    print(generated_text)
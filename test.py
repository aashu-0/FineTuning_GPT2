from load_weights import convert_gpt2_weights_to_custom
import tiktoken
import torch
from utils import text_to_token_ids, token_ids_to_text, generate
from config import GPTConfig

def test_generation():
    # config to match gpt2 small

    config = GPTConfig(
        vocab_size=50257,
        context_length=1024,
        emb_dim=768,
        n_heads=12,
        n_layers=12,
        drop_rate=0.1,
        qkv_bias=True
    )
    
    # load model with pretrained weights
    print("Loading model with pre-trained weights...")
    model = convert_gpt2_weights_to_custom(config)

    tokenizer = tiktoken.get_encoding('gpt2')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    # generate some text from the model given a start context
    test_cases = [
        {
            "prompt": "Once upon a time",
            "max_tokens": 30,
            "temperature": 0.7,
            "top_k": 40
        },
        {
            "prompt": "I want to learn about",
            "max_tokens": 30,
            "temperature": 0.8,
            "top_k": 50
        },
        {
            "prompt": "1234",
            "max_tokens": 30,
            "temperature": 0.9,
            "top_k": 40
        }
    ]

    for i, test in enumerate(test_cases):
        print(f"\nTest {i+1}: '{test['prompt']}' with temp={test['temperature']}, top_k={test['top_k']}")

        output_ids = generate(
            model = model,
            idx = text_to_token_ids(test['prompt'], tokenizer).to(device),
            max_new_tokens =test['max_tokens'],
            context_size= config.context_length,
            eos_id= 50256,
            temp=test['temperature'],
            top_k=test['top_k'])
        
        generated_text = token_ids_to_text(output_ids, tokenizer)
        print(f"Generated{i}: {generated_text}")


if __name__ == '__main__':
    test_generation()
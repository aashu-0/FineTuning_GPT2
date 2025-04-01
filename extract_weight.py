import torch
from transformers import GPT2LMHeadModel
import argparse
import os

# extarct gpt2 weights from HF transformers

def extract_gpt2_weights(model_name="gpt2", output_path="gpt2_weights.pt"):

    print(f"Loading GPT-2 model: {model_name}")
    model = GPT2LMHeadModel.from_pretrained(model_name)

    # a structured weights dict
    structured_weights = {
        "embedding": {
            "token_embedding": model.transformer.wte.weight.clone(),
            "position_embedding": model.transformer.wpe.weight.clone()
        },
        "blocks": [],
        "final_norm": {
            "scale": model.transformer.ln_f.weight.clone(),
            "shift": model.transformer.ln_f.bias.clone()
        },
        "lm_head": model.lm_head.weight.clone()
    }

    # transformer blocks
    for i, block in enumerate(model.transformer.h):
        qkv_weight = block.attn.c_attn.weight.clone()
        qkv_bias = block.attn.c_attn.bias.clone()
        
        # embed dimension
        emb_dim = qkv_weight.size(0)
        
        # split qkv weighhts and biases
        # as in huggine face transformers, they're concatenated as [q, k, v]
        q_weight, k_weight, v_weight = qkv_weight.chunk(3, dim=1)
        q_bias, k_bias, v_bias = qkv_bias.chunk(3)

        block_weights = {
            "attention": {
                "query": {
                    "weight": q_weight.t(),  # transpose
                    "bias": q_bias
                },
                "key": {
                    "weight": k_weight.t(),  # transpose
                    "bias": k_bias
                },
                "value": {
                    "weight": v_weight.t(),  # transpose
                    "bias": v_bias
                },
                "out_proj": {
                    "weight": block.attn.c_proj.weight.clone(),
                    "bias": block.attn.c_proj.bias.clone()
                }
            },
            "norm1": {
                "scale": block.ln_1.weight.clone(),
                "shift": block.ln_1.bias.clone()
            },
            "norm2": {
                "scale": block.ln_2.weight.clone(),
                "shift": block.ln_2.bias.clone()
            },
            "ffn": {
                "fc1": {
                    "weight": block.mlp.c_fc.weight.clone(),
                    "bias": block.mlp.c_fc.bias.clone()
                },
                "fc2": {
                    "weight": block.mlp.c_proj.weight.clone(),
                    "bias": block.mlp.c_proj.bias.clone()
                }
            }
        }

        structured_weights["blocks"].append(block_weights)

    # save model config
    config = model.config
    structured_weights['config'] = {
        "vocab_size": config.vocab_size,
        "context_length": config.n_positions,
        "emb_dim": config.n_embd,
        "n_layers": config.n_layer,
        "n_heads": config.n_head,
        "drop_rate": config.resid_pdrop,
        "qkv_bias": True 
    }

    # save weights
    print(f'Saving weights in structured format to {output_path}')
    torch.save(structured_weights, output_path)

    # model summary
    print("\nGPT-2 Model Summary:")
    print(f"Model name: {model_name}")
    print(f"Vocabulary size: {config.vocab_size}")
    print(f"Context length: {config.n_positions}")
    print(f"Embedding dimension: {config.n_embd}")
    print(f"Number of layers: {config.n_layer}")
    print(f"Number of attention heads: {config.n_head}")

    return structured_weights


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract GPT-2 weights from HuggingFace")
    parser.add_argument("--model", type=str, default="gpt2", help="GPT-2 model name (e.g., gpt2, gpt2-medium)")
    parser.add_argument("--output", type=str, default="gpt2_weights.pt", help="Output file path")
    
    args = parser.parse_args()

    extract_gpt2_weights(args.model, args.output)
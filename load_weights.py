import torch
from model import GPTModel
from config import GPTConfig, get_device
import argparse
import os

def load_gpt2_weights(model, weights_path):
    print(f"Loading weights from {weights_path}")
    weights = torch.load(weights_path, map_location='cpu')
    
    # load embeddings
    model.tok_emb.weight.data.copy_(weights["embedding"]["token_embedding"])
    model.pos_emb.weight.data.copy_(weights["embedding"]["position_embedding"])

    # Load transformer blocks
    for i, block in enumerate(model.trf_blocks):
        # Load attention weights
        block.attn.W_query.weight.data.copy_(weights["blocks"][i]["attention"]["query"]["weight"])
        block.attn.W_key.weight.data.copy_(weights["blocks"][i]["attention"]["key"]["weight"])
        block.attn.W_value.weight.data.copy_(weights["blocks"][i]["attention"]["value"]["weight"])
        
        # Load bias if present in our model
        if hasattr(block.attn.W_query, 'bias') and block.attn.W_query.bias is not None:
            block.attn.W_query.bias.data.copy_(weights["blocks"][i]["attention"]["query"]["bias"])
            block.attn.W_key.bias.data.copy_(weights["blocks"][i]["attention"]["key"]["bias"])
            block.attn.W_value.bias.data.copy_(weights["blocks"][i]["attention"]["value"]["bias"])
        
        # Load output projection
        block.attn.out_proj.weight.data.copy_(weights["blocks"][i]["attention"]["out_proj"]["weight"])
        if hasattr(block.attn.out_proj, 'bias') and block.attn.out_proj.bias is not None:
            block.attn.out_proj.bias.data.copy_(weights["blocks"][i]["attention"]["out_proj"]["bias"])
        
        # Load layer norm weights
        block.norm1.scale.data.copy_(weights["blocks"][i]["norm1"]["scale"])
        block.norm1.shift.data.copy_(weights["blocks"][i]["norm1"]["shift"])
        block.norm2.scale.data.copy_(weights["blocks"][i]["norm2"]["scale"])
        block.norm2.shift.data.copy_(weights["blocks"][i]["norm2"]["shift"])
        
        # Load FFN weights
        block.ff.layers[0].weight.data.copy_(weights["blocks"][i]["ffn"]["fc1"]["weight"])
        block.ff.layers[0].bias.data.copy_(weights["blocks"][i]["ffn"]["fc1"]["bias"])
        block.ff.layers[2].weight.data.copy_(weights["blocks"][i]["ffn"]["fc2"]["weight"])
        block.ff.layers[2].bias.data.copy_(weights["blocks"][i]["ffn"]["fc2"]["bias"])
    
    # Load final layer norm
    model.final_norm.scale.data.copy_(weights["final_norm"]["scale"])
    model.final_norm.shift.data.copy_(weights["final_norm"]["shift"])
    
    # Load LM head
    model.out_head.weight.data.copy_(weights["lm_head"])
    
    print("Successfully loaded GPT-2 weights into custom model")
    return model


def create_model_from_pretrained(weights_path=None, model_size='small'):

    # If weights are provided, load the config from them
    if weights_path and os.path.exists(weights_path):
        config = GPTConfig.from_pretrained(weights_path)
    else:
        # Otherwise, use the specified config
        config = GPTConfig.from_model_size(model_size)
    
    # Create the model
    model = GPTModel(config)
    
    # Load weights if provided
    if weights_path and os.path.exists(weights_path):
        model = load_gpt2_weights(model, weights_path)
    
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load GPT-2 weights into custom model")
    parser.add_argument("--weights", type=str, default="gpt2_weights.pt", help="Path to extracted weights")
    parser.add_argument("--model_size", type=str, default="small", help="Model size if not using weights")
    parser.add_argument("--output", type=str, default=None, help="Path to save the loaded model (optional)")
    
    args = parser.parse_args()
    
    # Create model with pretrained weights
    model = create_model_from_pretrained(args.weights, args.model_size)
    model = model.to(get_device())
    
    # Save the model if requested
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
        torch.save(model.state_dict(), args.output)
        print(f"Model saved to {args.output}")
    
    print("Model loaded successfully")
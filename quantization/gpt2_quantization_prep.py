import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.quantization import QuantStub, DeQuantStub
from typing import Optional
from base_model.config import GPT2Config


class LoRALayer(nn.Module):
    """Low-Rank Adaptation layer for efficient fine-tuning."""
    
    def __init__(self, in_dim: int, out_dim: int, rank: int, alpha: float):
        super().__init__()
        self.A = nn.Parameter(torch.empty(in_dim, rank))
        torch.nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.B = nn.Parameter(torch.zeros(rank, out_dim))
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.alpha * (x @ self.A @ self.B)


class LinearWithLoRA(nn.Module):
    """Linear layer combined with LoRA adaptation."""
    
    def __init__(self, linear: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(
            linear.in_features,
            linear.out_features,
            rank,
            alpha
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) + self.lora(x)


class CausalMultiHeadAttention(nn.Module):
    """Multi-head attention with quantization stubs for post-training quantization."""
    
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.n_heads = config.n_heads
        self.emb_dim = config.emb_dim
        self.head_dim = self.emb_dim // self.n_heads

        assert self.emb_dim % self.n_heads == 0, "Embedding dimension must be divisible by number of heads"

        # Core attention layers
        self.qkv = nn.Linear(self.emb_dim, 3 * self.emb_dim)
        self.out_proj = nn.Linear(self.emb_dim, self.emb_dim)
        
        # Dropout layers
        self.attn_dropout_p = config.dropout_attn
        self.res_dropout = nn.Dropout(config.dropout_res)

        # Quantization stubs for post-training quantization
        self.input_quant = QuantStub()
        self.input_dequant = DeQuantStub()
        self.output_quant = QuantStub()
        self.output_dequant = DeQuantStub()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, num_tokens, c = x.size()

        # Quantize input for QKV computation
        x_quant = self.input_quant(x)
        qkv = self.qkv(x_quant)
        qkv = self.input_dequant(qkv)

        # Split into Q, K, V and reshape for multi-head attention
        q, k, v = qkv.split(self.emb_dim, dim=2)
        k = k.view(b, num_tokens, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        q = q.view(b, num_tokens, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(b, num_tokens, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        # Efficient scaled dot-product attention
        weights = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=True,
            dropout_p=self.attn_dropout_p if self.training else 0.0
        )

        # Reshape and combine heads
        output = weights.transpose(1, 2).contiguous().view(b, num_tokens, self.emb_dim)

        # Quantize output projection
        output = self.output_quant(output)
        output = self.out_proj(output)
        output = self.output_dequant(output)
        
        # Apply residual dropout
        output = self.res_dropout(output)
        return output


class MLP(nn.Module):
    """MLP block with quantization stubs."""
    
    def __init__(self, config: GPT2Config):
        super().__init__()
        
        # Quantization stubs
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

        # MLP layers
        self.fc1 = nn.Linear(config.emb_dim, 4 * config.emb_dim)
        self.gelu = nn.GELU(approximate='tanh')
        self.fc2 = nn.Linear(4 * config.emb_dim, config.emb_dim)
        self.dropout = nn.Dropout(config.dropout_res)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.quant(x)
        x = self.dropout(self.fc2(self.gelu(self.fc1(x))))
        x = self.dequant(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer block with layer normalization, attention, and MLP."""
    
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.emb_dim)
        self.attn = CausalMultiHeadAttention(config)
        self.ln2 = nn.LayerNorm(config.emb_dim)
        self.mlp = MLP(config)

        # Quantization stubs for residual connections
        self.residual_quant = QuantStub()
        self.residual_dequant = DeQuantStub()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm attention block with residual connection
        attn_input = self.ln1(x)
        attn_input = self.residual_quant(attn_input)
        attn_output = self.attn(attn_input)
        attn_output = self.residual_dequant(attn_output)
        x = x + attn_output

        # Pre-norm MLP block with residual connection
        x = x + self.mlp(self.ln2(x))
        return x


class GPTModelQuantized(nn.Module):
    """GPT2 model with quantization stubs for post-training quantization."""
    
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.config = config

        # Embedding layers
        self.tok_emb = nn.Embedding(config.vocab_size, config.emb_dim)
        self.pos_emb = nn.Embedding(config.context_length, config.emb_dim)
        self.emb_dropout = nn.Dropout(config.dropout_emb)

        # Main quantization stubs
        self.main_quant = QuantStub()
        self.main_dequant = DeQuantStub()

        # Transformer blocks
        self.trf_blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])

        # Output layers
        self.final_norm = nn.LayerNorm(config.emb_dim)
        self.out_head = nn.Linear(config.emb_dim, config.vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        _, current_seq_len = input_ids.size()
        
        # Create position embeddings
        pos_embs = self.pos_emb(torch.arange(current_seq_len, device=input_ids.device))

        # Combine token and position embeddings
        x = self.tok_emb(input_ids) + pos_embs
        x = self.emb_dropout(x)

        # Quantize before transformer blocks
        x = self.main_quant(x)
        
        # Pass through transformer blocks
        for block in self.trf_blocks:
            x = block(x)

        # Final layer normalization
        x = self.final_norm(x)

        # Dequantize before output head
        x = self.main_dequant(x)
        logits = self.out_head(x)
        
        return logits


def replace_linear_with_lora(model: nn.Module, rank: int, alpha: float) -> None:
    """
    Recursively replace all Linear layers in the model with LinearWithLoRA layers.
    
    Args:
        model: The model to modify
        rank: LoRA rank parameter
        alpha: LoRA alpha scaling parameter
    """
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            # Replace Linear layer with LinearWithLoRA
            setattr(model, name, LinearWithLoRA(module, rank, alpha))
        else:
            # Recursively apply to child modules
            replace_linear_with_lora(module, rank, alpha)


def count_parameters(model: nn.Module, only_trainable: bool = True) -> int:
    """Count the number of parameters in the model."""
    if only_trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())

def freeze_base_parameters(model: nn.Module) -> None:
    """Freeze all parameters in the base model (before LoRA injection)."""
    for param in model.parameters():
        param.requires_grad = False


def prepare_model_for_quantization(
    config: Optional[GPT2Config] = None,
    lora_rank: int = 16,
    lora_alpha: float = 16,
    model_path: Optional[str] = None
) -> GPTModelQuantized:
    """
    Prepare GPT2 model for post-training quantization with LoRA adaptation.
    
    Args:
        config: Model configuration (uses default if None)
        lora_rank: LoRA rank parameter
        lora_alpha: LoRA alpha scaling parameter  
        model_path: Path to pre-trained model weights (optional)
        
    Returns:
        Prepared model with quantization stubs and LoRA layers
    """
    if config is None:
        config = GPT2Config()

    print("Initializing GPT2 model with quantization stubs...")
    model = GPTModelQuantized(config)
    
    # Print initial parameter count
    total_params = count_parameters(model, only_trainable=False)
    trainable_params = count_parameters(model, only_trainable=True)
    print(f"Initial total parameters: {total_params:,}")
    print(f"Initial trainable parameters: {trainable_params:,}")

    # Freeze base model parameters
    print("\nFreezing base model parameters...")
    freeze_base_parameters(model)
    trainable_params = count_parameters(model, only_trainable=True)
    print(f"Trainable parameters after freezing: {trainable_params:,}")

    # Inject LoRA layers
    print(f"\nInjecting LoRA layers (rank={lora_rank}, alpha={lora_alpha})...")
    replace_linear_with_lora(model, rank=lora_rank, alpha=lora_alpha)
    trainable_params = count_parameters(model, only_trainable=True)
    print(f"Trainable LoRA parameters: {trainable_params:,}")

    # Load pre-trained weights if provided
    if model_path and os.path.exists(model_path):
        print(f"\nLoading pre-trained weights from: {model_path}")
        try:
            state_dict = torch.load(model_path, map_location='cpu')
            model.load_state_dict(state_dict)
            print("Successfully loaded pre-trained weights!")
        except Exception as e:
            print(f"Warning: Failed to load weights: {e}")
    elif model_path:
        print(f"Warning: Model path {model_path} does not exist")

    print("\nModel preparation complete!")
    print("The model is now ready for post-training quantization.")
    
    return model


def main():
    """Main function to demonstrate model preparation."""
    
    # Initialize configuration
    config = GPT2Config()
    
    # Prepare model (adjust model_path as needed)
    model = prepare_model_for_quantization(
        config=config,
        lora_rank=16,
        lora_alpha=16,
        model_path="quantization/gpt2_lorafinetuned.pt"  # Set to your model path if available
    )
    
    # Example: Test model forward pass
    print("\nTesting model forward pass...")
    batch_size, seq_len = 2, 10
    dummy_input = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)
        print(f"Output shape: {output.shape}")
        print(f"Expected shape: ({batch_size}, {seq_len}, {config.vocab_size})")

if __name__ == "__main__":
    main()
# GPT-2: From Scratch & Fine-Tuning on Alpaca

A minimal yet effective implementation of GPT-2 in PyTorch. This project loads pretrained GPT-2 weights and enables text generation while providing insights into the model’s underlying architecture.
It includes fine-tuning capabilities on the Stanford Alpaca dataset for instruction following. (not fine tuned yet)

![GPT-2 Architecture](assets/image.png)  
*Source: [ResearchGate](https://www.researchgate.net/figure/GPT-2-model-architecture-The-GPT-2-model-contains-N-Transformer-decoder-blocks-as-shown_fig1_373352176)*


### Base Model
- `base_model/model.py`: Core GPT-2 model implementation
- `base_model/MHA.py`: Multi-head attention with flash attention for speed
- `base_model/TransformerBlock.py`: Transformer block implementation with pre-layernorm
- `base_model/config.py`: Configuration settings matching gpt2-small (12 layers, 768 dim, 12 heads)
- `base_model/load_weights.py`: Maps Hugging Face GPT-2 weights to this implementation
- `base_model/utils.py`: Helper functions for tokenization and text generation
- `base_model/test.py`: Test the model with different prompts

### Fine-tuning
- `fine_tune/dataset.py`: Dataset loading and processing for Stanford Alpaca
- `fine_tune/train.py`: Training loop and evaluation functions
- `fine_tune/utils.py`: Formatting functions for instruction data and visualization tools


## setup

```bash
# Create venv
python -m venv aienv
source aienv/bin/activate  # or .\aienv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
```

## running the base model

```bash
# Quick test with pretrained weights
python base_model/test.py
```

<!-- ## Fine-tuning on Alpaca Dataset

```bash
# Download and prepare the dataset
python fine_tune/dataset.py

# Fine-tune the model
python fine_tune/train.py
``` -->

## how it works

### base model
1. loads official Hugging Face GPT-2 weights
2. maps them to our custom implementation 
3. uses tiktoken for tokenization
4. generates text with temperature and top-k sampling

<!-- ### Fine-tuning
1. downloads and preprocesses the Stanford Alpaca dataset
2. formats inputs as instruction-following examples
3. implements a training loop with gradient accumulation
4. provides evaluation and sample generation during training -->

## Examples

### base model
try your own prompts by modifying `base_model/test.py` or importing the model directly:

```python
from base_model.load_weights import load_gpt2_weights_to_model
from base_model.utils import text_to_token_ids, token_ids_to_text, generate
from base_model.config import GPT2Config
import tiktoken

config = GPT2Config()
model = load_gpt2_weights_to_model(config)
tokenizer = tiktoken.get_encoding('gpt2')

output_ids = generate(
    model=model,
    idx=text_to_token_ids("your prompt here", tokenizer),
    max_new_tokens=30,
    context_size=config.context_length,
    temp=0.7,
    top_k=40
)

print(token_ids_to_text(output_ids, tokenizer))
```

## soon
- instruction fine-tuning gpt2 on Stanford alpaca dataset
- implementing lora for efficient training
- explore various optimization techniques. 

## acknowledgments and references

- [Hugging Face's GPT-2 Implementation](https://huggingface.co/gpt2)
- [Build a Large Language Model (From Scratch) Book](https://github.com/rasbt/LLMs-from-scratch)
- [Andrej Karpathy’s Let's reproduce GPT-2 ](https://youtu.be/l8pRSuU81PU?si=vELvndsmquwRzyB9)
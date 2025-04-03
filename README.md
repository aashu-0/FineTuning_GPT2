# gpt2-from-scratch

minimal implementation of gpt2 in pytorch. this code loads pretrained gpt2 weights and lets you generate text. doesn't reinvent the wheel, just shows how the architecture works under the hood.

![GPT2 architecture](image.png)
*architecture diagram of gpt2 model implemented in this repo. source: [image source](https://www.researchgate.net/figure/GPT-2-model-architecture-The-GPT-2-model-contains-N-Transformer-decoder-blocks-as-shown_fig1_373352176)*

## what's in the box

- `model.py`: core gpt2 model implementation
- `MHA.py`: multi-head attention with flash attention for speed
- `TransformerBlock.py`: transformer block implementation with pre-layernorm
- `config.py`: configuration settings matching gpt2-small (12 layers, 768 dim, 12 heads)
- `load_weights.py`: maps huggingface gpt2 weights to this implementation
- `utils.py`: helper functions for tokenization and text generation
- `test.py`: test the model with different prompts

## setup

```bash
# create venv
python -m venv aienv
source aienv/bin/activate  # or .\aienv\Scripts\activate on windows

# install deps
pip install -r requirements.txt
```

## run it

```bash
# quick test with pretrained weights
python test.py
```

## how it works

1. loads official huggingface gpt2 weights
2. maps them to our custom implementation 
3. uses tiktoken for tokenization
4. generates text with temperature and top-k sampling

## examples

try your own prompts by modifying `test.py` or importing the model directly:

```python
from load_weights import load_gpt2_weights_to_model
from utils import text_to_token_ids, token_ids_to_text, generate
from config import GPT2Config
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

## will add soon

- instruction fine-tuning on alpaca dataset
- lora, and various optimization techniques.
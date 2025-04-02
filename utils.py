import torch

# function to text to token_id and token_ids_to_text
def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special= {'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    decoded_text = tokenizer.decode(token_ids.squeeze(0).tolist())
    return decoded_text



# function to generate text
def generate(model,idx, max_new_tokens, context_size, temp=0.0,
             top_k = None, eos_id = None):
  
  device = idx.device
  model = model.to(device)
  # eval mode
  model.eval()

  for _ in range(max_new_tokens):
        # crops the current context(initial tokens) to fit model's max context size
        idx_cond = idx[:,-context_size:]

        with torch.no_grad():
            logits = model(idx_cond)  # shape (batch, n_token, vocab_size)

        logits = logits[:, -1, :]  # to extracts the last vector, shape -> (batch, vocab_size)

        # apply temperature
        if temp >0:
           logits = logits/temp

        # apply top-k filtering
        if top_k is not None and top_k >0:
          v, _= torch.topk(logits, min(top_k, logits.shape[-1]))
          logits[logits < v[:,[-1]]] = float('-inf')

        # logits to probas
        probs = torch.softmax(logits, dim=-1)

        # sample
        if temp ==0:
          # greddy sampling
          idx_next = torch.argmax(probs, dim=-1, keepdim=True)
        else:
          # temp scaling
          idx_next = torch.multinomial(probs, num_samples=1)

        # append new token to seq
        idx = torch.cat((idx, idx_next), dim=1)

        # stop when encounter eos token
        if eos_id is not None and idx_next == eos_id:
          break

  return idx
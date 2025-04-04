import torch
import torch.nn.functional as F
from base_model.utils import text_to_token_ids, token_ids_to_text, generate

def train_model_with_samples(
        model, train_dataloader, val_dataloader, optimizer,
        device, n_epochs, eval_freq, eval_iter, start_context,
        tokenizer,
        context_size= 1024,
        num_samples=3, sample_length=50, 
        grad_accum_steps=4):
    '''
    args ->
        model
        train_dataloader
        val_dataloader
        optimizer
        device
        n_epochs
        eval_freq: frequency (in steps) to evaluate model
        eval_iter: number of batches to use for eval
        start_context: starting text
        context_size
        tokenizer
        num_samples=3: number of text samples to generate
        sample_length=50
        grad_accum_steps=4: number of steps to accumlate gradients
        '''
    
    train_losses, val_losses, track_tokens_seen =[],[],[]
    token_seen, global_step = 0,0

    print(f'Training model for {n_epochs} epochs with gradient accumlation every {grad_accum_steps} steps')

    for epoch in range(n_epochs):
        model.train()
        epoch_loss =0
        accum_loss = 0

        print(f"\n{'='*50}")
        print(f"Epoch {epoch+1}/{n_epochs}")
        print(f"{'='*50}")

        for step, (input_batch, target_batch) in enumerate(train_dataloader):
            loss = calc_loss_batch(input_batch, target_batch,
                                   model, device)
            norm_loss = loss /grad_accum_steps
            # backward pass
            norm_loss.backward

            accum_loss += norm_loss.item()
            epoch_loss += norm_loss.item()

            # update params only after accumulating enough gradients
            if (step+1)% grad_accum_steps ==0 or step ==len(train_dataloader)-1:
                optimizer.step()
                optimizer.zero_grad()
                global_step +=1

                # track token seen (actual batch_size* grad_accum_steps)
                token_seen += input_batch.numel()* grad_accum_steps

                # eval step
                if global_step % eval_freq ==0:
                    train_loss, val_loss = evaluate_model(model, train_dataloader,
                                                          val_dataloader, device,
                                                          eval_iter)
                    train_losses.append(train_loss)
                    val_losses.append(val_losses)
                    track_tokens_seen.append(token_seen)

                    print(f'Epoch: {epoch+1}/{n_epochs} | Step: {global_step:06d} | Tokens: {token_seen:,}')
                    print(f'Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')

                accum_loss = 0
        # average epoch loss
        avg_epoch_loss = epoch_loss/ len(train_dataloader)
        print(f"\nEpoch {epoch+1} completed with average loss: {avg_epoch_loss:.4f}")

        # generate text samples after each epoch
        print(f"\n--- Generating {num_samples} text samples ---")
        model.eval()
        with torch.no_grad():
            for i in range(num_samples):
                print(f"\nSample {i+1}:")
                encoded = text_to_token_ids(start_context, tokenizer).to(device)
                token_ids = generate(model, idx=encoded,
                                    max_new_tokens=sample_length,
                                    context_size=context_size)
                decoded_text = token_ids_to_text(token_ids, tokenizer)
                print(f"Context: {start_context}")
                print(f"Generated: {decoded_text}")

        model.train() # back to training mode

    print(f"\nTraining completed. Total steps: {global_step}, Total tokens seen: {tokens_seen:,}")
    return train_losses, val_losses, track_tokens_seen


# loss calculation for a batch
def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device) # shape [batch_size, seq_len]
    target_batch = target_batch.to(device) # shape [batch_size, seq_len]

    #forward pass
    logits = model(input_batch)

    # calculate cross entropy loss
    # reshape logits from [batch_size, seq_len, vocab_size] -> [batch_size*seq_len, vocab_size]
    # reshape target_batch from [batch_size, seq_len] -> [batch_size*seq_len]

    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target_batch.view(-1))
    # or
    #loss = F.cross_entropy(logits.flatten(0, 1), target_batch.flatten()) # .view is perferred why? -> feel more standard 

    return loss


# calculate average loss over multiple batch from a dataloader
def calc_loss_loader(dataloader, model, device, num_batches=None):
    total_loss =0.0

    # if empty dataloader
    if len(dataloader) ==0:
        return float('nan')
    
    # num_batches
    if num_batches is None:
        num_batches = len(dataloader)
    else:
        num_batches = min(num_batches, len(dataloader))

    # accumulate losses over num_batches
    for i, (input_batch, target_batch) in enumerate(dataloader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss +=loss.item()
        else:
            break
    # avg loss
    return total_loss/num_batches


# eval function for both train and val data
def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    # eval mode
    eval.mode()

    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)

    # back to train
    model.train()

    return train_loss, val_loss
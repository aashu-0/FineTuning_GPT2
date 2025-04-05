import torch
import torch.nn.functional as F
from torch.optim import AdamW
from base_model.utils import text_to_token_ids, token_ids_to_text, generate
import wandb
from fine_tune.config import TrainingConfig


def train_model_with_samples(
        model, train_dataloader, val_dataloader, optimizer,
        device, tokenizer, config: TrainingConfig):
    
    #init wandb
    wandb.init(project= config.wandb_project,
               entity= config.wandb_entity,
               config=vars(config))

    train_losses, val_losses, track_tokens_seen =[],[],[]
    token_seen, global_step = 0,0

    print(f'Training model for {config.num_epochs} epochs with gradient accumlation every {config.grad_accum_steps} steps')

    for epoch in range(config.num_epochs):
        model.train()
        epoch_loss =0
        accum_loss = 0

        print(f"\n{'='*50}")
        print(f"Epoch {epoch+1}/{config.num_epochs}")
        print(f"{'='*50}")

        for step, (input_batch, target_batch) in enumerate(train_dataloader):
            loss = calc_loss_batch(input_batch, target_batch,
                                   model, device)
            norm_loss = loss /config.grad_accum_steps
            # backward pass
            norm_loss.backward()

            accum_loss += norm_loss.item()
            epoch_loss += norm_loss.item()

            # update params only after accumulating enough gradients
            if (step+1)% config.grad_accum_steps ==0 or step ==len(train_dataloader)-1:
                optimizer.step()
                optimizer.zero_grad()
                global_step +=1

                # track token seen (actual batch_size* grad_accum_steps)
                token_seen += input_batch.numel()* config.grad_accum_steps

                # eval step
                if global_step % config.eval_freq ==0:
                    train_loss, val_loss = evaluate_model(model, train_dataloader,
                                                          val_dataloader, device,
                                                          config.eval_iter)
                    train_losses.append(train_loss)
                    val_losses.append(val_loss)
                    track_tokens_seen.append(token_seen)

                    print(f'Epoch: {epoch+1}/{config.num_epochs} | Step: {global_step:06d} | Tokens: {token_seen:,}')
                    print(f'Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')

                    wandb.log({
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "tokens_seen": token_seen,
                        "global_step": global_step,
                        "epoch": epoch + 1
                    })

                accum_loss = 0
        # average epoch loss
        avg_epoch_loss = epoch_loss/ len(train_dataloader)
        print(f"\nEpoch {epoch+1} completed with average loss: {avg_epoch_loss:.4f}")

        # generate text samples after each epoch
        print(f"\n--- Generating {config.num_samples} text samples ---")
        model.eval()
        with torch.no_grad():
            samples = []
            for i in range(config.num_samples):
                print(f"\nSample {i+1}:")
                encoded = text_to_token_ids(config.start_context, tokenizer).to(device)
                token_ids = generate(model, idx=encoded,
                                    max_new_tokens=config.sample_length,
                                    context_size=config.context_length)
                decoded_text = token_ids_to_text(token_ids, tokenizer)
                print(f"Context: {config.start_context}")
                print(f"Generated: {decoded_text}")
                samples.append(decoded_text)

            wandb.log({
                "epoch": epoch + 1,
                "samples": wandb.Html("\n".join([f"<p>Sample {i+1}: {s}</p>" for i, s in enumerate(samples)]))
            })

        model.train() # back to training mode

    print(f"\nTraining completed. Total steps: {global_step}, Total tokens seen: {token_seen:,}")
    wandb.finish()
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
    model.eval()

    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)

    # back to train
    model.train()

    return train_loss, val_loss

# if __name__ == '__main__':
# -------------------------
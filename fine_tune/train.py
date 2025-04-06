import torch
import torch.nn.functional as F
from base_model.utils import text_to_token_ids, token_ids_to_text, generate
import wandb
from fine_tune.config import TrainingConfig


def train_model_with_samples(
        model, train_dataloader, val_dataloader, optimizer,
        start_context,lr_scheduler,
        device, tokenizer, config: TrainingConfig):

    model = model.to(device)

    #init wandb
    wandb.init(project= config.wandb_project,
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
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)

            #forward pass
            logits = model(input_batch)

            #loss
            loss = calc_loss_batch(logits, target_batch,
                                   device)
            norm_loss = loss /config.grad_accum_steps
            # backward pass
            norm_loss.backward()

            accum_loss += norm_loss.item()
            epoch_loss += norm_loss.item()

            # update params only after accumulating enough gradients
            if (step+1)% config.grad_accum_steps ==0 or step ==len(train_dataloader)-1:
                optimizer.step()
                lr_scheduler.step()
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
        print(f"\n--- Generating text sample ---")

        generate_and_print_sample(model, tokenizer, device, start_context, config)

    print(f"\nTraining completed. Total steps: {global_step}, Total tokens seen: {token_seen:,}")
    wandb.finish()
    return train_losses, val_losses, track_tokens_seen

def generate_and_print_sample(model, tokenizer, device, start_context, config: TrainingConfig):
    model.eval()
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate(model, idx=encoded,
                            max_new_tokens=config.sample_length,
                            context_size=config.context_length,
                             temp=0.7,
                             top_k=50)
        decoded_text = token_ids_to_text(token_ids.cpu(), tokenizer)
        decoded_text = decoded_text.replace('\n', ' ')
        
        print(f"Output: {decoded_text}")
    model.train()

# loss calculation for a batch
def calc_loss_batch(logits, target_batch, device):
    # input_batch = input_batch.to(device) # shape [batch_size, seq_len]
    # target_batch = target_batch.to(device) # shape [batch_size, seq_len]

    #forward pass
    # logits = model(input_batch)

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
            #tensor
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)

            #forward
            logits = model(input_batch)

            loss = calc_loss_batch(logits, target_batch, device)
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

if __name__ == '__main__':
    import tiktoken
    import torch
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR
    from base_model.load_weights import load_gpt2_weights_to_model
    from base_model.config import GPT2Config
    from fine_tune.config import TrainingConfig
    from fine_tune.dataset import download_dataset, load_subset, train_test_split, create_dataloader
    from transformers import get_scheduler

    # Initialize configs
    gpt2_config = GPT2Config()

    train_config = TrainingConfig(
    num_epochs =1,
    grad_accum_steps=2,
    batch_size=4,
    wandb_project = 'fine_tuning_gpt2_alpaca3k',
    subset_size =1000)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load tokenizer
    tokenizer = tiktoken.get_encoding('gpt2')
    
    # Load model with pretrained weights
    model = load_gpt2_weights_to_model(gpt2_config)
    model = model.to(device)
    
    # Download and prepare dataset
    full_dataset = download_dataset(train_config)
    subset_data = load_subset(full_dataset, train_config)
    train_data, test_data, val_data = train_test_split(subset_data, train_config)
    
    # Create dataloaders
    train_loader, test_loader, val_loader = create_dataloader(
        train_data, test_data, val_data, tokenizer, train_config, device=device
    )
    
    # Initialize optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.00005,
        weight_decay=0.1)
    num_training_steps = train_config.num_epochs * len(train_loader)
    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=int(0.1 * num_training_steps),
        num_training_steps=num_training_steps)
    
    # Sample prompt for generation during training
    start_context = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nExplain the concept of deep learning in simple terms.\n\n### Response:\n"
    
    # Train the model
    train_losses, val_losses, track_tokens_seen = train_model_with_samples(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        start_context=start_context,
        device=device,
        tokenizer=tokenizer,
        config=train_config
    )
    
    # Save the fine-tuned model
    torch.save(model.state_dict(), "gpt2_finetuned.pt")
    print("Fine-tuned model saved to 'gpt2_finetuned.pt'")
    
    # Plot training results
    from fine_tune.utils import plot_losses
    # Create epochs array (assuming eval_freq steps per epoch for plotting)
    epochs = [i * train_config.eval_freq / (len(train_loader) / train_config.grad_accum_steps) for i in range(len(train_losses))]
    plot_losses(epochs, track_tokens_seen, train_losses, val_losses)
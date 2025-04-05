from dataclasses import dataclass

@dataclass
class TrainingConfig:
    # training params
    num_epochs = 3
    grad_accum_steps =4
    eval_freq = 5
    eval_iter=5
    learning_rate=5e-5
    
    # dataset params
    url= "https://raw.githubusercontent.com/gururise/AlpacaDataCleaned/refs/heads/main/alpaca_data_cleaned.json"
    file_name = "alpaca_data.json"
    subset_file_name = "subset_file.json"
    subset_size =3000
    seed =1234
    train_ratio= 0.85
    test_ratio=0.1
    batch_size =8
    allowed_max_length =512
    
    # sampling params
    context_length =512
    sample_length = 80

    # wandb params
    wandb_project = "gpt2-finetuning"
    wandb_entity = ""
    # log_interval = 100
    
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
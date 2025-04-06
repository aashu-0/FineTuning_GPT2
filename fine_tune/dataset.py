# instruction fine tuning our custom gpt2 on alpaca dataset from stanford
import random
import json
import urllib.request
import torch
from torch.utils.data import Dataset, DataLoader
from fine_tune.utils import format_input
from fine_tune.config import TrainingConfig

# download the dataset
def download_dataset(config: TrainingConfig):
    urllib.request.urlretrieve(config.url, config.file_name)

    # load
    with open(config.file_name, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    print('Data Loaded Successfully')
    print(f'Number of entries in dataset: {len(dataset)}')
    return dataset


# extract a subset from full dataset (optional)
def load_subset(dataset, config: TrainingConfig):
    random.seed(config.seed)
    subset_data = random.sample(dataset, config.subset_size) # -> returns a list of dictionaries

    # list to json-formatted string
    with open(config.subset_file_name, 'w', encoding='utf-8') as f:
        json.dump(subset_data, f, indent=4)

    # load
    with open(config.subset_file_name, 'r', encoding='utf-8') as f:
        subset_dataset = json.load(f)
    print('Subset Data loaded successfully')
    print(f'Number of entries in subset dataset: {len(subset_data)}')
    return subset_dataset


# train-test split
def train_test_split(dataset, config: TrainingConfig):
    '''
    returns train_data, test_data, val_data
    '''

    random.seed(config.seed)

    total_size = len(dataset)
    train_size = int(total_size * config.train_ratio)
    val_size = int(total_size * config.val_ratio)

    train_data = dataset[:train_size]
    val_data = dataset[train_size: train_size+val_size]
    test_data = dataset[train_size+val_size: ]

    print(f'Train set size: {len(train_data)}')
    print(f'Test set size: {len(test_data)}')
    print(f'Validation set size: {len(val_data)}')
    return train_data, test_data, val_data


# custom dataset class
class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.encode_texts = []
        for entry in data:
            input_with_instruction = format_input(entry)
            response_text = f"{entry['output']}"
            full_text = input_with_instruction + response_text

            self.encode_texts.append(tokenizer.encode(full_text))
    def __getitem__(self, index):
        return self.encode_texts[index]

    def __len__(self):
        return len(self.data)


# custom batch collate function
# why not deafult_collate??
# we want to do some custom preprocessing which includes
# 1. adjusting the input ids to have same length (using <pad> token)
#2. create target token ids (input ids shifted by 1)
#3. replace certain(all except first) <pad> tokens in target ids with -100 to exclude them from training loss

def custom_collate_fn(batch, config: TrainingConfig,
                      pad_token_id = 50256, 
                      ignore_index=-100, device = None):
    
    # find the longest seq in the batch and then pads entire batch upto that length
    batch_max_length = max(len(item)+1 for item in batch)
    inputs_lst =[]
    targets_lst = []

    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id]
        padded_input = new_item + [pad_token_id]*(batch_max_length- len(new_item))

        inputs = torch.tensor(padded_input[:-1]) # remove the extra pad tok added eariler
        targets = torch.tensor(padded_input[1:]) # shift by 1

        # replacing all <pad> except first by -100
        mask = targets == pad_token_id

        indices = torch.nonzero(mask).squeeze() # it returns the indices of true values
        if indices.numel() >1:
            targets[indices[1:]] = ignore_index

        # truncating the tensors to allowed max length is not None
        if config.allowed_max_length is not None:
            inputs = inputs[:config.allowed_max_length]
            targets = targets[:config.allowed_max_length]

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    # stack the tensors
    inputs_tensor = torch.stack(inputs_lst)
    targets_tensor = torch.stack(targets_lst)
    
    # move to device if specified
    if device is not None:
        inputs_tensor = inputs_tensor.to(device)
        targets_tensor = targets_tensor.to(device)
    return inputs_tensor, targets_tensor


# custom dataloader
def create_dataloader(train_data, test_data, val_data,
                      tokenizer, config: TrainingConfig, device= None):
    train_dataset = InstructionDataset(train_data, tokenizer)
    test_dataset = InstructionDataset(test_data, tokenizer)
    val_dataset = InstructionDataset(val_data, tokenizer)

    # collate function with the correct device
    def collate_with_device(batch):
        return custom_collate_fn(
            batch, 
            config,
            pad_token_id=50256, 
            ignore_index=-100, 
            device=device
        )

    # dataloaders
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                              shuffle=True, collate_fn=collate_with_device)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size,
                              shuffle=False, collate_fn=collate_with_device)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size,
                              shuffle=False, collate_fn=collate_with_device)
    
    return train_loader, test_loader, val_loader


if __name__ == '__main__':
    import tiktoken

    config = TrainingConfig()

    full_dataset = download_dataset(config)
    sub_dataset = load_subset(full_dataset, config)
    example = sub_dataset[random.randint(0,len(sub_dataset))]
    formatted_ex = format_input(example)

    print(f'Random Example:\n{formatted_ex}\n')

    # split
    train_data, test_data, val_data = train_test_split(sub_dataset,config)

    tokenizer = tiktoken.get_encoding('gpt2')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    #dataloaders
    train_loader, test_loader, val_loader = create_dataloader(
        train_data, test_data, val_data, tokenizer, config, device=device
    )

    for idx, (X, y) in enumerate(train_loader):
        print(f'Input Shape: {X.shape} | Target Shape: {y.shape}')
        break
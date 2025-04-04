# instruction fine tuning our custom gpt2 on alpaca dataset from stanford

import random
import json
import urllib.request
import torch
from torch.utils.data import Dataset, DataLoader
from utils import format_input

# download the dataset
def download_dataset(url, file_name):
    urllib.request.urlretrieve(url, file_name)

    # load
    with open(file_name, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    print('Data Loaded Successfully')
    print(f'Number of entries in dataset: {len(dataset)}')
    return dataset


# extract a subset from full dataset (optional)
def load_subset(dataset, subset_file_name, subset_size):
    random.seed(123)
    subset_data = random.sample(dataset, subset_size) # -> returns a list of dictionaries

    # list to json-formatted string
    with open(subset_file_name, 'w', encoding='utf-8') as f:
        json.dump(subset_data, f, indent=4)

    # load
    with open(subset_file_name, 'r', encoding='utf-8') as f:
        subset_dataset = json.load(f)
    print('Subset Data loaded successfully')
    print(f'Number of entries in subset dataset: {len(subset_data)}')
    return subset_dataset


# train-test split
def train_test_split(dataset, train_fraction, test_fraction):
    train_set = int(len(dataset)*0.85)
    test_set = int(len(dataset)*0.1)
    val_test = len(dataset) - train_set -test_set

    train_data = dataset[:train_set]
    test_data = dataset[train_set: train_set+test_set]
    val_data = dataset[train_set+test_set:]

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
            response_text = f'{entry['output']}'
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

def custom_collate_fn(batch, pad_token_id = 50256, ignore_index=-100,
                      allowed_max_length= 512, device=torch.device()):
    
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
        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    # stack the tensors
    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)
    return inputs_tensor, targets_tensor


# Dataloader class
# ---------------------------

if __name__ == '__main__':
    url = "https://raw.githubusercontent.com/gururise/AlpacaDataCleaned/refs/heads/main/alpaca_data_cleaned.json"
    file_name = 'alpaca_data.json'

    full_dataset = download_dataset(url, file_name)
    print(f'Random Example\n: {full_dataset[random.randint(0,len(full_dataset))]}')
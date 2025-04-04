# formatting input text
# there are various ways to format instruction entries, but we will stick with alpaca prompt style

import random

def format_input(entry):
    # extract components
    instruction = entry.get('instruction', '')
    input_text = entry.get('input', '')
    # output = entry.get('output', '')

    # formatting
    prompt = 'Below is an instruction that describes a task'

    if input_text:
        prompt += ', paired with an input that provides further context'

    prompt += ".\n\n"
    prompt += "Write a response that appropriately completes the request.\n\n"
    prompt += f"### Instruction:\n{instruction}\n\n"

    if input_text:
        prompt += f"### Input:\n{input_text}\n\n"
    prompt += "### Response:\n"

    # if output:
    #     prompt += f'{output}'
    return prompt


if __name__ == '__main__':
    from dataset import download_dataset
    url = "https://raw.githubusercontent.com/gururise/AlpacaDataCleaned/refs/heads/main/alpaca_data_cleaned.json"
    file_name = 'alpaca_data.json'
    full_dataset = full_dataset = download_dataset(url, file_name)
    example = full_dataset[random.randint(0,len(full_dataset))]

    formatted_ex = format_input(example)
    print(formatted_ex)
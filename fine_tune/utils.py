import random
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# formatting input text
# there are various ways to format instruction entries, but we will stick with alpaca prompt style
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


# Plot Training and Validation losses
def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    fig, ax1 = plt.subplots(figsize =(5,3))
    ax1.plot(epochs_seen, train_losses, label= 'Training Loss')
    ax1.plot(epochs_seen, val_losses, linestyle ='-.', label = 'Validation Loss')

    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend(loc= 'upper right')

    ax1.xaxis.set_major_locator(MaxNLocator(integer = True))
    ax2 = ax1.twiny() # creates a twin x-axis (shared y-axis) for existing axis ax1

    ax2.plot(tokens_seen, train_losses, alpha=0) # alpha = 0 -> invisible plot
    ax2.set_xlabel('Tokens seen')
    fig.tight_layout()
    plt.show()


# main
if __name__ == '__main__':
    from dataset import download_dataset
    url = "https://raw.githubusercontent.com/gururise/AlpacaDataCleaned/refs/heads/main/alpaca_data_cleaned.json"
    file_name = 'alpaca_data.json'
    full_dataset = full_dataset = download_dataset(url, file_name)
    example = full_dataset[random.randint(0,len(full_dataset))]

    formatted_ex = format_input(example)
    print(formatted_ex)
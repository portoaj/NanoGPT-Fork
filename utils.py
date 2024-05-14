from typing import List, Union
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

with open('input.txt', 'r') as input_file:
    text = input_file.read()

chars = sorted(list(set(text)))

stoi = { ch: i for i, ch in enumerate(chars)}
itos = { i: ch for i, ch in enumerate(chars)}

def encode(text: str)->List[int]:
    return [stoi[char] for char in text]


def decode(tokens: List[int])->str:
    return ''.join([itos[token] for token in tokens])

data = torch.tensor(encode(text), dtype=torch.long)

# Create train test split
split = int(len(text) * 0.9)
train_data = data[:split]
val_data = data[split:]

torch.manual_seed(0)

# Pass in the batch split (train or val)
# Get a batch of data back

batch_size = 4
block_size = 8 # Max context length

def get_text():
    return text

def get_vocab():
    return chars

def get_batch(batch_size: int, split: str):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i+1:i + block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


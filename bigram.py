from utils import *
import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(0)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
vocab_size = len(get_vocab())
xb, yb = get_batch(4, 'train')
print('inputs', xb.shape, xb)
print('targets', yb.shape, yb)

n_embed = 32

class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.language_modeling_head = nn.Linear(n_embed, vocab_size)
    
    def forward(self, idx, targets=None):
        # index and targets are both (B, T) tensors of integers
        # where B is batch num and T is block size
        token_embeddings = self.token_embedding_table(idx) # B, T, C (C= n_embed?)
        logits = self.language_modeling_head(token_embeddings) # (B, T, vocab_size)
        # C is channels and in this case is vocab_size
        if targets is None:
            loss = None
        else:
            # Reshape logits for cross entropy
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, loss = self(idx)
            # Get only logits from the last tiem step
            logits = logits[:, -1, :] # becomes (B, C)
            # Get the probability for each token
            probs = F.softmax(logits, dim=1) # (B, C)
            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # Append sampled index to running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx


model = BigramLanguageModel()
m = model.to(device)

logits, loss = m(xb, yb)
print(logits.shape)
print(loss)

idx = torch.zeros((1,1), dtype=torch.long, device=device)
print(decode(m.generate(idx, max_new_tokens=100)[0].tolist()))

# Create a pytorch optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

# train the model
print(device)
BATCH_SIZE = 32
EPOCHS = 4501
for steps in range(EPOCHS):
    # Get a batch of data
    xb, yb = get_batch(BATCH_SIZE, 'train')

    # Evaluate the loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    if steps % 500 == 0:
        print(loss.item())

print(decode(m.generate(idx, max_new_tokens=100)[0].tolist()))
import torch
from torch.nn import functional as F
import torch.nn as nn

B, T, C = 4, 8, 32
x = torch.randn(B,T,C)
# xbow = torch.zeros((B, T, C))

# for b in range(B):
#     for t in range(T):
#         xprev = x[b,:t+1]
#         xbow[b, t] = torch.mean(xprev, 0)

# #print(x[0])
# #print(xbow[0])

tril = torch.tril(torch.ones(T, T))
# wei = torch.zeros((T,T))
# wei = wei.masked_fill(tril == 0, float('-inf'))
# wei = F.softmax(wei, dim=1)
# xbow3 = wei @ x
#print(torch.allclose(xbow, xbow3))
#wei = wei / wei.sum(1, keepdim=True)
#print(wei)

#xbow2 = wei @ x # ((B dimension inferred here by pytorch as well to keep dimensions equal)T, T) @ (B, T, C) -> (B, T,C)
#print(torch.allclose(xbow, xbow2))

head_size = 16
key = nn.Linear(C, head_size, bias=False)
query = nn.Linear(C, head_size, bias=False)
value = nn.Linear(C, head_size, bias=False)

k = key(x) # (B, T, head_size)
q = query(x) # (B, T, head_size)

wei = q @ k.transpose(-2, -1) * head_size ** -.5 # (B, T, head_size) @ (B, head_size, T) -> (B, T, T)
wei = wei.masked_fill(tril == 0, float('-inf'))
wei = F.softmax(wei, dim=-1)

v = value(x)
out = wei @ v
print(wei[0])

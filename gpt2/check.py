import torch
from model import GPTConfig, GPT

torch.manual_seed(0)
model = GPT(GPTConfig(block_size=16, n_layer=2, n_head=4, n_embd=64)).eval()
start = torch.zeros((2, 1), dtype=torch.long)
out = model.generate(start, max_new_tokens=20)
print("생성 결과 모양:", out.shape)
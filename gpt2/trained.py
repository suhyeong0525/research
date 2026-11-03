import os
import urllib.request

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tiktoken
import torch

from model import GPT, GPTConfig

torch.manual_seed(1337)
device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
print("device:", device)

# 1) 데이터 준비
path = "data/input.txt"
if not os.path.exists(path):
    os.makedirs("data", exist_ok=True)
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    urllib.request.urlretrieve(url, path)
text = open(path, encoding="utf-8").read()
enc = tiktoken.get_encoding("gpt2")
data = torch.tensor(enc.encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]
print(f"토큰 수: 학습 {len(train_data):,} / 검증 {len(val_data):,}")

# 2) 설정
block_size, batch_size = 128, 16
max_iters, eval_interval, eval_iters = 2000, 100, 20
lr = 6e-4
config = GPTConfig(block_size=block_size, n_layer=4, n_head=4, n_embd=256)
model = GPT(config).to(device)
print(f"파라미터 수: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)


def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - block_size - 1, (batch_size,))
    x = torch.stack([d[i:i + block_size] for i in ix])
    y = torch.stack([d[i + 1:i + block_size + 1] for i in ix])   # 정답 = 한 칸 뒤
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss():
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = torch.zeros(eval_iters)
        for i in range(eval_iters):
            x, y = get_batch(split)
            _, loss = model(x, y)
            losses[i] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


# 3) 학습
steps, train_hist, val_hist = [], [], []
for it in range(max_iters + 1):
    if it % eval_interval == 0:
        l = estimate_loss()
        steps.append(it)
        train_hist.append(l["train"])
        val_hist.append(l["val"])
        print(f"step {it:5d} | train {l['train']:.3f} | val {l['val']:.3f}")
    if it == max_iters:
        break
    x, y = get_batch("train")
    _, loss = model(x, y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# 4) 결과 저장
os.makedirs("../results", exist_ok=True)
plt.plot(steps, train_hist, label="train")
plt.plot(steps, val_hist, label="val")
plt.xlabel("step")
plt.ylabel("loss")
plt.legend()
plt.title("GPT-2 (small config) on tiny Shakespeare")
plt.savefig("../results/gpt2_loss.png", dpi=150)

model.eval()
start = torch.tensor([enc.encode("ROMEO:")] * 3, device=device)
out = model.generate(start, max_new_tokens=100, top_k=50)
with open("../results/gpt2_samples.txt", "w", encoding="utf-8") as f:
    for row in out:
        f.write(enc.decode(row.tolist()) + "\n\n-----\n\n")
print("저장 완료: ../results/gpt2_loss.png, ../results/gpt2_samples.txt")
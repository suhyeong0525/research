import torch
import tiktoken
from transformers import GPT2LMHeadModel

from model import GPT, GPTConfig
from load_pretrained import load_hf_weights

hf = GPT2LMHeadModel.from_pretrained("gpt2").float().eval()
model = load_hf_weights(GPT(GPTConfig()), hf).eval()

enc = tiktoken.get_encoding("gpt2")
ids = torch.tensor([enc.encode("Hello, I'm a language model,")])

with torch.no_grad():
    my_logits, _ = model(ids)
    hf_logits = hf(ids).logits

diff = (my_logits - hf_logits).abs().max().item()
print(f"최대 차이: {diff:.2e}")
assert diff < 1e-3, "logits가 다릅니다. 레이어 이름, GELU, 마스크를 확인하세요."
print("검증 통과: 내 GPT-2와 HF GPT-2의 출력이 같습니다.")

torch.manual_seed(42)
out = model.generate(ids.repeat(3, 1), max_new_tokens=30, top_k=50)
for row in out:
    print("---")
    print(enc.decode(row.tolist()))
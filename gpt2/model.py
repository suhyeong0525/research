import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class GPTConfig:
    block_size: int = 1024        # 한 번에 입력 가능한 토큰 수 -> 이 놈이 위치 가중치 크기랑도 같음
    vocab_size: int = 50257  # 토큰 종류 수
    n_layer: int = 12        # 블록 개수 -> 몇 개의 층으로 쌓을 건지. 
    n_head: int = 12         # 어텐션 헤드 수 -> 어텐션을 몇 개로 나눠서 역할을 부여할 것인지. 
    n_embd: int = 768        # 토큰 벡터 차원 -> 하나의 토큰을 몇 차원의 벡터로 임베딩할 것인지.

class CausalSelfAttention(nn.Module): # 상속자와 부모.
    def __init__(self,config):        # 생성자
        super().__init__()            # 부모 생성자의 호출
        assert config.n_embd % config.n_head == 0 # 임베딩 된 넘버를 어텐션 헤드 수로 나눌 수 있는지 확인한다. 
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.c_attn = nn.Linear(config.n_embd, 3*config.n_embd) # 768칸의 백터 받아서 2304칸 뱉어내는 것. 
        self.c_proj = nn.Linear(config.n_embd, config.n_embd) # 12개 섞은 다음에 합치는 놈이니까 768 * 768이다. 
        mask = torch.tril(torch.ones(config.block_size, config.block_size)) # 앞 단어가 뒷 단어 못 보게 마스크 만들기 1024*1024 사이즈로 만들고 위에 부분 0으로 만들기 
        self.register_buffer("mask", mask.view(1, 1, config.block_size, config.block_size), persistent=False) #모든 문장 모든 헤드에 같은 것 적용. 

    def forward(self, x): # 입력 x가 들어왔을 때의 계산 입력은 (B, T, C) 배치 크기(문장), 문장당 토큰 수, 벡터 차원
        B,T,C = x.size()
        hs = C // self.n_head # 어텐션 나누는 박스 크기 정함. 
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2) # 공부한 예시로 보면 (2,8,768) 을 c_attn 이용해서 3배 만든다. (2,8,2304) 그리고 dim=2 즉 2304를 n_embd 크기로 자른다. 
        q = q.view(B, T, self.n_head, hs).transpose(1, 2)  # (B, nh, T, hs) 768을 12*64로 나누고 T와 nh의 위치를 바꾼다. 
        k = k.view(B, T, self.n_head, hs).transpose(1, 2)
        v = v.view(B, T, self.n_head, hs).transpose(1, 2)        
        att = (q @ k.transpose(-2, -1)) / math.sqrt(hs)     # (B, nh, T, T) 이제 개념 공부할 때 나온 점수 계산 들어간다. k의 역행렬 만들어서 q랑 내적 크기의 제곱으로 나눈다. 
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf")) # 소프트맥스 들어가기 전에 마스크 씌워서 0인 애들은 음의 무한대 만들어야 소프트맥스 이후에 0이 된다.  -> 어차피 지금 들어온 토큰 수까지만의 크기만 필요하니까 T,T로 자르는 것. 
        att = F.softmax(att, dim=-1)
        y = att @ v                                          # (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)     # 자리 바꾸고 헤드 이어 붙이기 다시 (B,T,C) 크기가 된다. 
        return self.c_proj(y)

class MLP(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd) # 사이즈 4배하고 
        self.gelu = nn.GELU(approximate="tanh") # GELU 하고
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd) # 사이즈 원래대로 돌라오고. 

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict
        self.transformer = nn.ModuleDict(dict( # 딕셔너리 만들기 Embedding 자료형의 wte 
            wte=nn.Embedding(config.vocab_size, config.n_embd), # 토큰 임베딩한 표 만들기 
            wpe=nn.Embedding(config.block_size, config.n_embd), # 위치 임베딩한 표 만들기
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]), # 높이 12짜리인 블럭들 생성. 12번 반복해야 한다. 
            ln_f=nn.LayerNorm(config.n_embd), # 12개의 박스를 거치고 난 뒤 마지막 정규화를 위한 층.
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False) # 최종 벡터를 768 크기를 50257 로 바꿔서 다음에 올 가장 가능성 높은 단어를 선택한다. 입력과 출력이 같은 가중치를 사용한다. 
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None): #idx에는 (2,8) 처럼 각 문장과 토큰 수의 크기이고 값으로는 0부터 50256 사이의 값이 들어가 있다. 
        B,T = idx.size()
        assert T <= self.config.block_size, "입력이 block_size보다 깁니다"
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x=self.transformer.wte(idx)+self.transformer.wpe(pos)
        for block in self.transformer.h:
            x=block(x)
        x=self.transformer.ln_f(x)
        logits =self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad() # 이 함수 안에서는 기울기 연산 하지 마라 메모리 덜 쓴다. 
    def generate(self, idx, max_new_tokens, top_k=50, temperature=1.0): # idx 시작 문장의 토큰 번호  max 는 몇 개를 만들지
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:] # 맨 뒤에서 1024 잘라서 가져온다. 
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            k = min(top_k, logits.size(-1))
            v, _ = torch.topk(logits, k)
            logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_id), dim=1)
        return idx




        
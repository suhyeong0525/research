import torch

TRANSPOSED = ("attn.c_attn.weight", "attn.c_proj.weight", "mlp.c_fc.weight", "mlp.c_proj.weight")


def load_hf_weights(model, hf_model):
    """HF GPT2LMHeadModel의 가중치를 내 GPT 모델에 복사한다."""
    sd = model.state_dict()
    sd_hf = hf_model.state_dict()
    # HF가 Conv1D를 쓰면 c_attn 가중치 모양이 (768, 2304)로, 내 Linear (2304, 768)와 뒤집혀 있다
    n_embd = model.config.n_embd
    conv1d = tuple(sd_hf["transformer.h.0.attn.c_attn.weight"].shape) == (n_embd, 3 * n_embd)
    for k in sd:
        src = sd_hf.get(k)
        if src is None and k == "lm_head.weight":
            src = sd_hf["transformer.wte.weight"]   # 공유 가중치라 이름이 하나만 있을 수 있음
        if src is None:
            raise KeyError(f"HF 쪽에 없는 키: {k}")
        if conv1d and k.endswith(TRANSPOSED):
            src = src.t()
        if src.shape != sd[k].shape:
            raise ValueError(f"모양 불일치 {k}: 내 모델 {tuple(sd[k].shape)} vs HF {tuple(src.shape)}")
        with torch.no_grad():
            sd[k].copy_(src)
    return model
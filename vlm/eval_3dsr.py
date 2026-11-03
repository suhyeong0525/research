"""3DSRBench synthetic 평가: 행마다 모델의 답 글자를 JSONL로 저장한다.

사용 예:
  python vlm/eval_3dsr.py --out /content/drive/MyDrive/research/results/baseline_synth.jsonl --limit 200
  python vlm/eval_3dsr.py --out .../sat5k_synth.jsonl --adapter /content/drive/MyDrive/research/adapters/sat5k
"""
import argparse
import json
import os
import re
import time
import zipfile

import torch
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import AutoModelForMultimodalLM, AutoProcessor

DATASET = "wufeim/3DSRBench_synthetic"   # 표에는 파일 이름(image_url)만 있고, 사진은 images.zip에 따로 있음

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="Qwen/Qwen3.5-2B")
parser.add_argument("--adapter", default=None, help="LoRA 어댑터 폴더 (파인튜닝 후 평가할 때만)")
parser.add_argument("--out", required=True, help="결과 JSONL 경로")
parser.add_argument("--limit", type=int, default=0, help="앞에서 몇 행만 평가 (0이면 전체)")
parser.add_argument("--image_dir", default="/content/3dsr_images", help="images.zip을 풀어 둘 폴더")
args = parser.parse_args()


def prepare_images(image_dir):
    """images.zip을 한 번만 받아 풀고, {파일 이름: 전체 경로} 표를 만든다."""
    if not os.path.isdir(image_dir) or not os.listdir(image_dir):
        print("images.zip 내려받는 중...")
        zpath = hf_hub_download(DATASET, "images.zip", repo_type="dataset")
        with zipfile.ZipFile(zpath) as z:
            z.extractall(image_dir)
    paths = {}
    for root, _, names in os.walk(image_dir):   # zip 안 폴더 구조와 상관없이 파일 이름으로 찾기
        for n in names:
            paths[n] = os.path.join(root, n)
    return paths


# 1) 데이터
ds = load_dataset(DATASET, split="test")
if args.limit:
    ds = ds.select(range(min(args.limit, len(ds))))
image_paths = prepare_images(args.image_dir)
missing = [r for r in ds["image_url"] if r not in image_paths]
print(f"평가할 행 수: {len(ds)} | 사진 {len(image_paths)}장 | 못 찾은 사진 {len(missing)}개")
if missing:
    raise SystemExit(f"사진을 못 찾았습니다. 예: {missing[:3]}")

# 2) 이미 끝낸 행은 건너뛰기 (Colab이 끊겨도 이어서 실행 가능)
done = set()
if os.path.exists(args.out):
    with open(args.out, encoding="utf-8") as f:
        for line in f:
            done.add(json.loads(line)["index"])
    print(f"이미 끝낸 행: {len(done)}")

# 3) 모델
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
processor = AutoProcessor.from_pretrained(args.model)
model = AutoModelForMultimodalLM.from_pretrained(args.model, dtype=dtype, device_map="auto")
if args.adapter:
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, args.adapter)
model.eval()


def build_prompt(row):
    letters = [k for k in "ABCD" if row.get(k) not in (None, "")]
    lines = [row["question"]] + [f"{k}. {row[k]}" for k in letters]
    lines.append("Answer with the option's letter from the given choices directly.")
    return "\n".join(lines), letters


def parse_letter(text, letters):
    t = text.strip()
    if t and t[0].upper() in letters:
        return t[0].upper()
    m = re.search(r"\b([A-D])\b", t.upper())
    if m and m.group(1) in letters:
        return m.group(1)
    return ""   # 글자를 못 찾으면 오답 처리


# 4) 평가
os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
start = time.time()
with open(args.out, "a", encoding="utf-8") as f:
    for i, row in enumerate(ds):
        if row["index"] in done:
            continue
        prompt, letters = build_prompt(row)
        messages = [{"role": "user", "content": [
            {"type": "image", "image": Image.open(image_paths[row["image_url"]]).convert("RGB")},
            {"type": "text", "text": prompt},
        ]}]
        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt", enable_thinking=False,
        ).to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=8, do_sample=False)
        text = processor.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        pred = parse_letter(text, letters)

        rec = {
            "index": row["index"], "qid": row["qid"], "category": row["category"],
            "viewpoint": row["viewpoint"], "answer": row["answer"],
            "pred": pred, "raw": text,
        }
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.flush()

        if i < 3 or i % 50 == 0:
            print(f"[{i}] {row['qid']} {row['viewpoint']} 정답 {row['answer']} 예측 {pred} ({text!r}) "
                  f"| {time.time() - start:.0f}초")

print("완료:", args.out)
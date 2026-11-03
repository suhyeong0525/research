"""평가 결과 JSONL로 정확도 표와 그래프를 만든다.

사용 예:
  python vlm/summarize.py baseline=/content/drive/MyDrive/research/results/baseline_synth.jsonl
  python vlm/summarize.py baseline=.../baseline_synth.jsonl lora=.../sat5k_synth.jsonl
"""
import json
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_groups(path):
    """같은 원문항(qid 앞부분) + 같은 시점의 변형을 하나로 묶는다."""
    groups = defaultdict(list)
    rows = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            base = r["qid"].split("-")[0]
            groups[(base, r["viewpoint"], r["category"])].append(r["pred"] == r["answer"])
            rows += 1
    return groups, rows


def accuracy(groups, cond=lambda key: True):
    """공식 규칙: 묶음 안의 변형을 모두 맞혀야 정답."""
    vals = [all(v) for k, v in groups.items() if cond(k)]
    return (100.0 * sum(vals) / len(vals), len(vals)) if vals else (float("nan"), 0)


runs = []
for arg in sys.argv[1:]:
    name, path = arg.split("=", 1)
    runs.append((name, *load_groups(path)))
if not runs:
    raise SystemExit("사용법: python summarize.py 이름=결과.jsonl [이름=결과.jsonl ...]")

categories = sorted({k[2] for _, g, _ in runs for k in g})

print("\n## 전체")
print("| 실험 | 평가 행 | 문항 수 | 전체 | common | uncommon | 격차 (common - uncommon) |")
print("| --- | --- | --- | --- | --- | --- | --- |")
summary = {}
for name, g, rows in runs:
    all_acc, n = accuracy(g)
    c, _ = accuracy(g, lambda k: k[1] == "common")
    u, _ = accuracy(g, lambda k: k[1] == "uncommon")
    summary[name] = (c, u)
    print(f"| {name} | {rows} | {n} | {all_acc:.1f} | {c:.1f} | {u:.1f} | {c - u:+.1f} |")

print("\n## 질문 유형별 (전체 시점)")
print("| 유형 | " + " | ".join(name for name, _, _ in runs) + " |")
print("| --- | " + " | ".join("---" for _ in runs) + " |")
for cat in categories:
    cells = []
    for _, g, _ in runs:
        a, n = accuracy(g, lambda k, cat=cat: k[2] == cat)
        cells.append(f"{a:.1f} (n={n})")
    print(f"| {cat} | " + " | ".join(cells) + " |")

# 그래프: 실험마다 common / uncommon 막대
out_dir = os.path.dirname(sys.argv[1].split("=", 1)[1]) or "."
fig, ax = plt.subplots(figsize=(5 + len(runs), 4))
width = 0.35
xs = range(len(runs))
ax.bar([x - width / 2 for x in xs], [summary[n][0] for n, _, _ in runs], width, label="common")
ax.bar([x + width / 2 for x in xs], [summary[n][1] for n, _, _ in runs], width, label="uncommon")
ax.set_xticks(list(xs))
ax.set_xticklabels([n for n, _, _ in runs])
ax.set_ylabel("Accuracy (%)")
ax.set_ylim(0, 100)
ax.set_title("3DSRBench synthetic: common vs uncommon viewpoints")
ax.legend()
fig.tight_layout()
png = os.path.join(out_dir, "viewpoint_bar.png")
fig.savefig(png, dpi=150)
print("\n그래프 저장:", png)
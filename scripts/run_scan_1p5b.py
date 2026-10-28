"""全量扫描 Qwen2.5-1.5B 的 196 个 Linear × (8,4,2) bit"""
import logging
import time
from pathlib import Path

import torch

logging.getLogger("bitsandbytes").setLevel(logging.ERROR)

from transformers import AutoModelForCausalLM, AutoTokenizer

from aq.metrics import load_text
from aq import sensitivity as SE

NAME = "Qwen/Qwen2.5-1.5B-Instruct"


def dtype_kw(d):
    import transformers
    maj = int(transformers.__version__.split(".")[0])
    return {"dtype": d} if maj >= 5 else {"torch_dtype": d}


tok = AutoTokenizer.from_pretrained(NAME)
model = AutoModelForCausalLM.from_pretrained(
    NAME, device_map={"": 0}, **dtype_kw(torch.bfloat16)).eval()

ids = tok(load_text("en"), return_tensors="pt").input_ids[0]
targets = SE.list_linear_layers(model)
print(f"目标 {len(targets)} 个 Linear，总参数 "
      f"{sum(t['params'] for t in targets):,}", flush=True)

Path("results").mkdir(exist_ok=True)
out = "results/sensitivity_en.csv"
Path(out).unlink(missing_ok=True)

t0 = time.perf_counter()
rows = SE.scan(model, tok, ids, bits_list=(8, 4, 2), targets=targets,
               out_csv=out, progress=False)
print(f"扫描完成 {len(rows)} 行，耗时 {time.perf_counter()-t0:.0f}s", flush=True)

# 汇总：最敏感 / 最不敏感
import csv
data = SE.from_csv(out)
by = {}
for r in data:
    by.setdefault(r["target"], {})[r["bits"]] = r["rel_pct"]

print("\n最敏感的 8 个（按 2bit 相对增幅）:")
for t, d in sorted(by.items(), key=lambda kv: -kv[1].get(2, -999))[:8]:
    print(f"  {t[-50:]:<50} 8bit {d.get(8,0):+7.2f}%  4bit {d.get(4,0):+7.2f}%  "
          f"2bit {d.get(2,0):+9.2f}%")

print("\n最不敏感的 8 个（按 2bit 相对增幅）:")
for t, d in sorted(by.items(), key=lambda kv: kv[1].get(2, 999))[:8]:
    print(f"  {t[-50:]:<50} 8bit {d.get(8,0):+7.2f}%  4bit {d.get(4,0):+7.2f}%  "
          f"2bit {d.get(2,0):+9.2f}%")

vals = [d.get(2, 0) for d in by.values()]
import statistics
print(f"\n2bit 相对增幅: min {min(vals):+.3f}%  中位 {statistics.median(vals):+.3f}%  "
      f"max {max(vals):+.3f}%")
print(f"极差 = {max(vals)-min(vals):.2f} 个百分点")
print(f"最大/中位绝对值比 = {max(abs(v) for v in vals)/max(abs(statistics.median(vals)),1e-9):.0f}x")
print("ALL_DONE", flush=True)

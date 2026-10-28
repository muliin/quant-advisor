"""四策略实测对照：均匀 / AlphaQ 式 / 实测式 / 混合

这是"把 α 当候选策略之一，在用户模型上实测比较"的具体实现。
每个策略都产出 {模块: bit} 的分配，然后用【真实 PPL】评估。
"""
import logging
import math
import statistics
import time

import numpy as np
import torch

logging.getLogger("bitsandbytes").setLevel(logging.ERROR)

from transformers import AutoModelForCausalLM, AutoTokenizer

from aq.allocate import Unit, size_gib
from aq.fakequant import quantize_model_inplace, restore
from aq.metrics import load_text, ppl_from_ids
from aq.sensitivity import from_csv
from aq.solve import pareto_front_mckp, solve_dp
from aq.spectral import eta_cost, load_spectral_csv, spectral_stats

NAME = "Qwen/Qwen2.5-1.5B-Instruct"
LM_HEAD = 0.434


def dtype_kw(d):
    import transformers
    maj = int(transformers.__version__.split(".")[0])
    return {"dtype": d} if maj >= 5 else {"torch_dtype": d}


print("=" * 76)
print("加载模型")
print("=" * 76)
tok = AutoTokenizer.from_pretrained(NAME)
model = AutoModelForCausalLM.from_pretrained(
    NAME, device_map={"": 0}, **dtype_kw(torch.bfloat16)).eval()
ids = tok(load_text("en"), return_tensors="pt").input_ids[0]
base, _ = ppl_from_ids(model, ids, chunk=512)
print(f"  基线 PPL = {base:.4f}")

rows = from_csv("results/sensitivity_en_nf4.csv")
sens = {r["target"]: r["rel_pct"] for r in rows if r["bits"] == 2}
sens_by_bits = {}
for r in rows:
    sens_by_bits.setdefault(r["target"], {})[r["bits"]] = r["rel_pct"]
params = {r["target"]: r["params"] for r in rows}
names = sorted(sens)
print(f"  {len(names)} 个模块（实测 ΔPPL 已有）")

# ============================================================ 谱指标
print("\n" + "=" * 76)
print("计算谱指标（α + 方差）")
print("=" * 76)
t0 = time.perf_counter()
sp = spectral_stats(model, use_pooled=True)
print(f"  {len(sp)} 个 Linear，耗时 {time.perf_counter()-t0:.0f}s")
alphas = np.array([sp[n]["alpha"] for n in names])
alpha_bar = float(np.median(alphas))
va = float(np.var(alphas))
gamma_def = float(alphas.min() * (alphas.max() - alphas.min()) / va)
print(f"  α: 中位 {alpha_bar:.4f}  范围 "
      f"{alphas.min():.3f}~{alphas.max():.3f}  γ_default {gamma_def:.2f}")

# ============================================================ 真实 PPL
def real_ppl(assign: dict) -> float:
    pname = {f"{k}.weight": v for k, v in assign.items()}
    h = quantize_model_inplace(model, pname, bits_default=16)
    try:
        v, _ = ppl_from_ids(model, ids, chunk=512)
    finally:
        restore(model, h)
    if ppl_from_ids(model, ids, chunk=512)[0] != base:
        raise RuntimeError("还原失败")
    return v


def gib_of(assign):
    return LM_HEAD + sum(size_gib(params[n], b) for n, b in assign.items())


# ============================================================ 四个策略
def s_uniform(bits=4):
    return {n: bits for n in names}


def s_alphaq(budget, cand=(2, 4, 8, 16)):
    """AlphaQ 式：η 作为代价，DP 求最优"""
    units = [Unit(name=n, params=params[n],
                  cost={b: eta_cost(params[n], sp[n]["var"], sp[n]["alpha"],
                                    alpha_bar, gamma_def, b)
                        for b in cand}) for n in names]
    r = solve_dp(units, budget, candidate_bits=cand,
                 fixed_gib=LM_HEAD, n_buckets=6000)
    return r["assign"]


def s_measured(budget, cand=(2, 4, 8, 16)):
    """实测式：用真实 ΔPPL 的 log 值（截断负值）作为代价"""
    units = []
    for n in names:
        # ★ log 空间代价（10.26：ΔPPL% 不可加，Δlog PPL 才近似可加）
        # ★★ 并且在 0 处截断：负的 ΔPPL 是"量化偶然有益"，不能当成收益，
        #    否则求解器会倾向于"压得越多越好"
        c = {16: 0.0}
        for b in (2, 4, 8):
            r = sens_by_bits.get(n, {}).get(b, sens.get(n, 0.0))
            c[b] = max(0.0, math.log1p(r / 100.0))
        units.append(Unit(name=n, params=params[n], cost=c))
    r = solve_dp(units, budget, candidate_bits=cand,
                 fixed_gib=LM_HEAD, n_buckets=6000)
    return r["assign"]


def s_hybrid(budget, top_k=24):
    """混合：top-K 最敏感（实测）用实测代价，其余用 α 的 η —— 一次 DP 求解"""
    keep = set(sorted(names, key=lambda n: -sens[n])[:top_k])
    units = []
    for n in names:
        if n in keep:
            c = {16: 0.0}
            for b in (2, 4, 8):
                r = sens_by_bits.get(n, {}).get(b, sens.get(n, 0.0))
                c[b] = max(0.0, math.log1p(r / 100.0))
        else:
            c = {b: eta_cost(params[n], sp[n]["var"], sp[n]["alpha"],
                             alpha_bar, gamma_def, b)
                 for b in (2, 4, 8, 16)}
        units.append(Unit(name=n, params=params[n], cost=c))
    r = solve_dp(units, budget, fixed_gib=LM_HEAD, n_buckets=6000)
    return r["assign"]


# ============================================================ 对照
print("\n" + "=" * 76)
print("四策略实测对照（预算 = 均匀 4bit 的显存）")
print("=" * 76)

b_uniform = gib_of(s_uniform(4))
print(f"  预算 = {b_uniform:.4f} GiB（均匀 4bit）\n")
print(f"  {'策略':<22}{'显存GiB':>10}{'真实PPL':>11}{'Δ%':>9}{'位宽分布':>22}")

results = {}
plans = {"均匀 4bit": s_uniform(4)}
for label, fn in (("AlphaQ 式（α，免校准）", s_alphaq),
                  ("实测式（ΔPPL，截断负值）", s_measured),
                  ("混合（实测 top24 + α）", s_hybrid)):
    try:
        plans[label] = fn(b_uniform)
    except Exception as e:
        print(f"  {label} 求解失败: {type(e).__name__}: {e}")
for label, a in plans.items():
    try:
        v = real_ppl(a)
    except Exception as e:
        print(f"  {label:<22}  失败: {type(e).__name__}: {e}")
        continue
    results[label] = (v, gib_of(a), a)
    hist = {}
    for x in a.values():
        hist[x] = hist.get(x, 0) + 1
    hs = " ".join(f"{k}b×{v2}" for k, v2 in sorted(hist.items()))
    print(f"  {label:<22}{gib_of(a):>10.4f}{v:>11.4f}"
          f"{(v/base-1)*100:>+8.2f}%{hs:>22}")

# ============================================================ 排序
print("\n" + "=" * 76)
print("排名（同预算下，ΔPPL 越小越好）")
print("=" * 76)
rank = sorted(results.items(), key=lambda kv: kv[1][0])
for i, (label, (v, g, _)) in enumerate(rank, 1):
    print(f"  {i}. {label:<24} ΔPPL {(v/base-1)*100:>+7.2f}%   "
          f"显存 {g:.4f} GiB")

print("\n" + "=" * 76)
print("补充：不同预算下 AlphaQ 式 vs 实测式（看排序会不会变）")
print("=" * 76)
print(f"  {'预算GiB':>9}{'AlphaQ式 Δ%':>15}{'实测式 Δ%':>13}{'谁更好':>10}")
for frac in (0.85, 1.00, 1.15, 1.30):
    b = b_uniform * frac
    try:
        v1 = real_ppl(s_alphaq(b))
    except Exception:
        v1 = float("nan")
    try:
        v2 = real_ppl(s_measured(b))
    except Exception:
        v2 = float("nan")
    better = ("实测式" if v2 < v1 else "AlphaQ式") if (v1 == v1 and v2 == v2) else "—"
    print(f"  {b:>9.3f}{(v1/base-1)*100:>+14.2f}%{(v2/base-1)*100:>+12.2f}%"
          f"{better:>10}")


print("\n" + "=" * 76)
print("追加：候选位宽只留 {4,8,16}（禁用 2bit 悬崖）")
print("=" * 76)
print(f"  {'策略':<26}{'显存GiB':>10}{'真实PPL':>11}{'Δ%':>9}{'分布':>20}")
cand = (4, 8, 16)
rows2 = {}
for label, fn in (("均匀 4bit", lambda b: s_uniform(4)),
                  ("AlphaQ 式 {4,8,16}", lambda b: s_alphaq(b, cand)),
                  ("实测式 {4,8,16}", lambda b: s_measured(b, cand))):
    try:
        a = fn(b_uniform)
        v = real_ppl(a)
    except Exception as e:
        print(f"  {label:<26}  失败 {type(e).__name__}")
        continue
    rows2[label] = v
    hist = {}
    for x in a.values():
        hist[x] = hist.get(x, 0) + 1
    hs = " ".join(f"{k}b×{v2}" for k, v2 in sorted(hist.items()))
    print(f"  {label:<26}{gib_of(a):>10.4f}{v:>11.4f}"
          f"{(v/base-1)*100:>+8.2f}%{hs:>20}")

if rows2:
    best = min(rows2.items(), key=lambda kv: kv[1])
    print(f"\n  ⇒ 最优: {best[0]}  ΔPPL {(best[1]/base-1)*100:+.2f}%")

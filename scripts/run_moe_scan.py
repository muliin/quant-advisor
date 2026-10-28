#!/usr/bin/env python
"""
run_moe_scan.py — MoE 逐专家敏感度扫描（云端自助运行）

用法（先冒烟，再全量）
    # ① 冒烟测试（2 分钟）：确认环境和模型都对
    python run_moe_scan.py --smoke

    # ② 全量：1024 个专家决策 × 2bit（约 35 分钟）
    python run_moe_scan.py --bits 2

    # ③ 补 4bit（可选，再 35 分钟）
    python run_moe_scan.py --bits 4

    # 中断了直接重跑同一条命令 —— 支持断点续跑

产出
    results/sensitivity_moe.csv   逐专家扫描结果（每专家 1 行）
    results/moe_scan_summary.txt  汇总（★ 把这个文件的内容贴回来）

依赖：torch, transformers, numpy（不需要 bitsandbytes）
模型：allenai/OLMoE-1B-7B-0924（首次会自动下载 12.9 GB）
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import sys
import time
from pathlib import Path

import torch

MODEL = "allenai/OLMoE-1B-7B-0924"
CSV_FIELDS = ["target", "kind", "bits", "dppl", "rel_pct", "params", "seconds"]


# ============================================================ 伪量化
def fake_quant(t: torch.Tensor, bits: int, group_size: int = 64,
               out_dtype: torch.dtype | None = None) -> torch.Tensor:
    """量化→立刻反量化。8bit+ 用 fp32 中间结果（bf16 尾数会成瓶颈）"""
    od = out_dtype or t.dtype
    x = t.detach().float()
    qmax = 2 ** (bits - 1) - 1

    flat = x.reshape(-1)
    n = flat.numel()
    pad = (-n) % group_size
    if pad:
        flat = torch.cat([flat, torch.zeros(pad, device=x.device)])
    blk = flat.reshape(-1, group_size)
    amax = blk.abs().amax(dim=1, keepdim=True)
    s = torch.where(amax > 0, amax / qmax, torch.full_like(amax, 1e-12))
    q = torch.clamp(torch.round(blk / s), -qmax - 1, qmax)
    return (q * s).reshape(-1)[:n].reshape(t.shape).to(od)


class Swap:
    """切一片参数 → 测 → 精确还原（用 torch.equal 断言）"""

    def __init__(self, param, new_data, index=None):
        self.param, self.new, self.index = param, new_data, index
        self.orig = None

    def __enter__(self):
        if self.index is None:
            self.orig = self.param.data.clone()
            self.param.data.copy_(self.new)
        else:
            self.orig = self.param.data[self.index].clone()
            self.param.data[self.index] = self.new
        return self

    def __exit__(self, *a):
        if self.index is None:
            self.param.data.copy_(self.orig)
            cur = self.param.data
        else:
            self.param.data[self.index] = self.orig
            cur = self.param.data[self.index]
        if not torch.equal(cur, self.orig):
            raise RuntimeError("★ 还原失败！测量结果不可信")
        return False


# ============================================================ PPL
EVAL_TEXT = """The history of artificial intelligence is usually dated from a summer
workshop held at Dartmouth College, although the ideas that converged there had been
circulating for more than a decade. Researchers believed that the central difficulty
had been identified, and that the remaining work was a matter of engineering. Programs
were written that could prove theorems from a small set of axioms, play checkers at a
respectable level, and manipulate blocks on a tabletop. Each success was reported as
evidence that general intelligence was close at hand. The collapse came when the funding
agencies asked for results outside those bounded worlds. A machine translation system
that performed acceptably on chemistry papers failed completely on political speeches.
A second period of enthusiasm arrived with expert systems. The insight behind them was
that a great deal of professional competence consists of rules of thumb accumulated
through experience, and that these rules could be elicited from practitioners and
encoded as conditional statements. Machine learning offered a different answer to the
same problem. Instead of asking an expert to articulate what they knew, one could
present examples and let a procedure infer a rule from them. This inverted the economics
of knowledge acquisition. Collecting examples was often cheaper than interviewing
experts, and the resulting models degraded gracefully when they encountered unfamiliar
inputs, producing a plausible answer rather than a confident contradiction. The attention
mechanism, introduced to improve translation, proved to be the key architectural idea.
Its function is to let each position in a sequence gather information from every other
position, with the amount gathered determined by a learned comparison between the two.
This has two advantages over sequential processing. It removes the requirement that
information travel one step at a time, so the distance between any two positions is a
single operation regardless of how far apart they are. And it allows the computation at
all positions to be performed simultaneously, which is what makes training on modern
hardware practical."""


def ppl_of(model, ids, chunk: int = 512):
    total_nll, total_tok = 0.0, 0
    for i in range(0, len(ids) - chunk, chunk):
        inp = ids[i:i + chunk].unsqueeze(0)
        with torch.no_grad():
            logits = model(inp).logits[0]
        sl, lb = logits[:-1, :], inp[0, 1:]
        assert sl.shape[0] == lb.shape[0] == inp.shape[1] - 1
        total_nll += torch.nn.functional.cross_entropy(
            sl.float(), lb, reduction="sum").item()
        total_tok += lb.numel()
    if total_tok == 0:
        raise ValueError("序列太短")
    import math
    return math.exp(total_nll / total_tok), total_tok


# ============================================================ 发现专家
def list_expert_tensors(model) -> list[dict]:
    """融合专家张量（3D Parameter）"""
    out = []
    for name, mod in model.named_modules():
        for pname, p in mod.named_parameters(recurse=False):
            if p.dim() == 3:
                out.append({"name": f"{name}.{pname}", "param": p,
                            "n_experts": p.shape[0], "shape": tuple(p.shape)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--bits", type=int, default=2)
    ap.add_argument("--group-size", type=int, default=64)
    ap.add_argument("--out", default="results/sensitivity_moe.csv")
    ap.add_argument("--summary", default="results/moe_scan_summary.txt")
    ap.add_argument("--smoke", action="store_true",
                    help="冒烟测试：1 层 × 2 专家 × 1 档")
    ap.add_argument("--max-tokens", type=int, default=1022)
    ap.add_argument("--layers", default="all",
                    help="要跑的层号，如 0,1,2,15；默认 all")
    ap.add_argument("--experts", default="all",
                    help="每层专家数（等间隔抽样）或 all")
    ap.add_argument("--fast", action="store_true",
                    help="快速模式：层 0,1,2,13,14,15 × 每层 16 专家")
    args = ap.parse_args()

    Path("results").mkdir(exist_ok=True)

    # ---------- 环境自检
    print("=" * 72)
    print("环境自检")
    print("=" * 72)
    print(f"  torch        {torch.__version__}")
    print(f"  CUDA 可用    {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        sys.exit("✗ 没有 CUDA，检查驱动")
    free, total = torch.cuda.mem_get_info()
    name = torch.cuda.get_device_name(0)
    print(f"  GPU          {name}")
    print(f"  显存         {free/1024**3:.2f} / {total/1024**3:.2f} GiB 可用")
    if total / 1024 ** 3 < 14:
        print(f"  ⚠️ 显存 {total/1024**3:.1f} GiB < 14 GiB，OLMoE 的 bf16（12.69 GiB）可能装不下")

    # ---------- 加载
    print("\n" + "=" * 72)
    print(f"加载 {args.model}（bf16）")
    print("=" * 72)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(args.model)
    kw = {}
    import transformers
    if int(transformers.__version__.split(".")[0]) < 5:
        kw["torch_dtype"] = torch.bfloat16
    else:
        kw["dtype"] = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        args.model, device_map={"": 0}, **kw).eval()
    print(f"  加载耗时 {time.perf_counter()-t0:.0f}s")
    print(f"  参数 {sum(p.numel() for p in model.parameters()):,}")
    print(f"  显存占用 {torch.cuda.memory_allocated()/1024**3:.2f} GiB")

    texts = []
    for f in ("data/en_eval.txt",):
        if Path(f).exists():
            texts.append(Path(f).read_text(encoding="utf-8"))
    ids = tok(texts[0] if texts else EVAL_TEXT,
              return_tensors="pt").input_ids[0][:args.max_tokens].to(model.device)
    print(f"  评测序列 {len(ids)} token")

    base, ntok = ppl_of(model, ids)
    print(f"  ★ 基线 PPL = {base:.4f}（{ntok} token 参与）")

    # ---------- 专家
    exps = list_expert_tensors(model)
    print(f"\n  找到 {len(exps)} 个融合专家张量：")
    for e in exps[:4]:
        print(f"    {e['name']:<44} {e['shape']} × {e['n_experts']} 专家")

    # ---------- 快速模式 / 过滤
    if args.fast:
        args.layers = "all"
        args.experts = "8"

    def _layer_of(name):
        return name.split(".layers.")[1].split(".")[0] if ".layers." in name \
            else "0"

    if args.layers != "all":
        want = {s.strip() for s in args.layers.split(",") if s.strip()}
        exps = [e for e in exps if _layer_of(e["name"]) in want]
        print(f"\n  【快速模式】只跑层 {sorted(want, key=int)}，"
              f"命中 {len(exps)} 个专家张量")

    def _pick_experts(n_total, spec):
        if spec == "all":
            return list(range(n_total))
        k = min(int(spec), n_total)
        # 等间隔抽样，可复现
        return [round(i * (n_total - 1) / max(k - 1, 1)) for i in range(k)]

    expert_sets = {}
    for e in exps:
        expert_sets[e["name"]] = _pick_experts(e["n_experts"], args.experts)
    if args.experts != "all":
        print(f"  每层抽 {args.experts} 个专家（等间隔）")

    n_layers = len({_layer_of(e["name"]) for e in exps})
    total_decisions = sum(len(expert_sets[e["name"]]) for e in exps)
    print(f"\n  层数 {n_layers}，决策点 {total_decisions} 个"
          f"（每个专家 × {len(exps)//max(n_layers,1)} 个张量）")

    # ---------- 冒烟
    if args.smoke:
        print("\n" + "=" * 72)
        print("冒烟测试：第 0 层 × 前 2 个专家 × 1 档")
        print("=" * 72)
        e0 = exps[0]
        p = e0["param"]
        for j in expert_sets[e0["name"]][:2]:
            t1 = time.perf_counter()
            with Swap(p, fake_quant(p.data[j], args.bits, args.group_size),
                      index=j):
                v, _ = ppl_of(model, ids)
            dt = time.perf_counter() - t1
            print(f"  {e0['name']}[{j}] {args.bits}bit → PPL {v:.4f} "
                  f"(Δ{(v/base-1)*100:+.2f}%)  耗时 {dt:.2f}s")
        print(f"\n  ⇒ 全量 {total_decisions} 个决策预计 "
              f"{total_decisions*dt/60:.0f} 分钟")
        print("  ⇒ 如果再跑 4bit，时间翻倍")
        print("\n  确认无误后运行：")
        print(f"    python {Path(sys.argv[0]).name} --bits {args.bits}")
        return

    # ---------- 断点续跑
    done = set()
    out = Path(args.out)
    if out.exists():
        for r in csv.DictReader(open(out, encoding="utf-8")):
            done.add((r["target"], int(r["bits"])))
        print(f"\n  断点续跑：已有 {len(done)} 条记录，会跳过")

    fh = open(out, "a", newline="", encoding="utf-8")
    wr = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
    if out.stat().st_size == 0:
        wr.writeheader()
        fh.flush()

    # ---------- 主循环
    print("\n" + "=" * 72)
    print(f"逐专家扫描（{args.bits}bit，group_size={args.group_size}）")
    print("=" * 72)
    t_start = time.perf_counter()
    n_done = 0
    for e in exps:
        p = e["param"]
        params_per_expert = p[0].numel()
        for j in expert_sets[e["name"]]:
            target = f"{e['name']}[{j}]"
            if (target, args.bits) in done:
                continue
            t1 = time.perf_counter()
            with Swap(p, fake_quant(p.data[j], args.bits, args.group_size),
                      index=j):
                v, _ = ppl_of(model, ids)
            d = v - base
            wr.writerow({"target": target, "kind": "expert",
                         "bits": args.bits, "dppl": round(d, 6),
                         "rel_pct": round(d / base * 100, 4),
                         "params": params_per_expert,
                         "seconds": round(time.perf_counter() - t1, 2)})
            fh.flush()
            n_done += 1
            if n_done % 20 == 0:
                el = time.perf_counter() - t_start
                eta = el / n_done * (total_decisions - n_done - len(done))
                print(f"  [{n_done}/{total_decisions - len(done)}] "
                      f"已用 {el/60:.0f} 分，剩余约 {eta/60:.0f} 分", flush=True)
    fh.close()
    print(f"\n  扫描完成，新增 {n_done} 条，用时 "
          f"{(time.perf_counter()-t_start)/60:.1f} 分钟")

    # ---------- 汇总
    print("\n" + "=" * 72)
    print("汇总")
    print("=" * 72)
    rows = list(csv.DictReader(open(out, encoding="utf-8")))
    rows = [r for r in rows if int(r["bits"]) == args.bits]
    vals = [(r["target"], float(r["rel_pct"])) for r in rows]
    vals.sort(key=lambda z: -z[1])
    if not vals:
        print("  没有数据")
        return
    v = [x[1] for x in vals]
    import statistics
    lines = [
        f"OLMoE 逐专家敏感度扫描（{args.bits}bit，group={args.group_size}）",
        f"基线 PPL = {base:.4f}",
        f"决策点数 = {len(vals)}",
        f"相对增幅: min {min(v):+.2f}%  中位 {statistics.median(v):+.2f}%  "
        f"max {max(v):+.2f}%",
        f"极差 = {max(v)-min(v):.2f} 个百分点",
        "",
        "最敏感的 10 个:",
    ]
    for t, x in vals[:10]:
        lines.append(f"  {t:<52} {x:+10.2f}%")
    lines.append("")
    lines.append("最不敏感的 10 个:")
    for t, x in vals[-10:][::-1]:
        lines.append(f"  {t:<52} {x:+10.2f}%")
    lines.append("")
    # 按层聚合
    by_layer = {}
    for t, x in vals:
        lay = t.split(".layers.")[1].split(".")[0] if ".layers." in t else "?"
        by_layer.setdefault(lay, []).append(x)
    lines.append("按层聚合（2bit 相对增幅的中位数）:")
    for lay in sorted(by_layer, key=int):
        med = statistics.median(by_layer[lay])
        lines.append(f"  layer {lay:>2}: 中位 {med:+10.2f}%  "
                     f"({len(by_layer[lay])} 个专家)")
    text = "\n".join(lines)
    Path(args.summary).write_text(text, encoding="utf-8")
    print(text)
    print(f"\n已写出 {out} 和 {args.summary}")
    print("★ 把 summary 的内容贴回来即可")


if __name__ == "__main__":
    main()

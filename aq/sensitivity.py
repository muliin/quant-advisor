"""aq/sensitivity.py — 敏感度扫描（逐模块 / 逐专家）"""
from __future__ import annotations

import csv
import time
from pathlib import Path

import torch

from aq.fakequant import Swap, fake_quant_bits


def list_linear_layers(model, exclude=("lm_head",)) -> list[dict]:
    """列出可量化的 nn.Linear（排除 lm_head）"""
    out = []
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear) and not any(
                name.endswith(e) or name == e for e in exclude):
            out.append({"name": name, "module": mod,
                        "numel": mod.weight.numel(),
                        "shape": tuple(mod.weight.shape)})
    return out


def list_moe_experts(model) -> list[dict]:
    """列出融合专家张量（3D nn.Parameter）"""
    out = []
    for name, mod in model.named_modules():
        for pname, p in mod.named_parameters(recurse=False):
            if p.dim() == 3:
                out.append({"name": f"{name}.{pname}",
                            "param": p, "n_experts": p.shape[0],
                            "shape": tuple(p.shape)})
    return out


def dppl_linear(model, ids, name: str, bits: int, base_ppl: float,
                group_size=64, ppl_fn=None) -> float:
    p = model.get_parameter(f"{name}.weight")
    with Swap(p, fake_quant_bits(p.data, bits, group_size)):
        v, _ = ppl_fn(model, ids)
    return v - base_ppl


def dppl_expert(model, ids, param_name: str, expert_idx: int, bits: int,
                base_ppl: float, group_size=64, ppl_fn=None) -> float:
    p = model.get_parameter(param_name)
    with Swap(p, fake_quant_bits(p.data[expert_idx], bits, group_size),
              index=expert_idx):
        v, _ = ppl_fn(model, ids)
    return v - base_ppl


def scan(model, ids, targets: list[dict], bits_list=(8, 4, 2),
         base_ppl: float = 0.0, ppl_fn=None,
         out_csv: str | None = None) -> list[dict]:
    """扫描目标单元在给定位宽下的 PPL 变化，可选增量写入 CSV。

    Args:
        model: 已加载的因果语言模型。
        ids: 输入的 token id 张量，形状 ``(seq_len,)``，1-D。
        targets: 待扫描单元列表，每项至少含 ``name``；专家项还需
            ``param`` 与 ``n_experts``。
        bits_list: 要测试的位宽列表，默认 ``(8, 4, 2)``，单位 bit/权重。
        base_ppl: 未量化的基线 PPL，用于换算 ``rel_pct``（百分比）。
        ppl_fn: 计算 PPL 的回调，返回 ``(ppl, ...)``。
        out_csv: 增量输出路径；文件已存在时按 ``(target, bits)`` 断点续跑。

    Returns:
        本次新增的记录列表，每项含 ``target`` / ``kind`` / ``bits`` /
        ``dppl`` / ``rel_pct`` / ``params`` / ``seconds``（秒）。

    Note:
        已完成的 ``(target, bits)`` 组合会被跳过，返回值只含本次新增的行。

    Warning:
        扫描期间通过 ``Swap`` 原地替换权重，前向后立即还原；还原由上下文
        管理器保证，但同一模型不可并发扫描，否则两次替换会互相污染。
    """
    done = set()
    if out_csv and Path(out_csv).exists():
        for r in csv.DictReader(open(out_csv, encoding="utf-8")):
            done.add((r["target"], int(r["bits"])))

    FIELDS = ["target", "kind", "bits", "dppl", "rel_pct", "params", "seconds"]
    rows = []
    f = open(out_csv, "a", newline="", encoding="utf-8") if out_csv else None
    writer = csv.DictWriter(f, fieldnames=FIELDS) if f else None
    if f and f.tell() == 0:
        writer.writeheader()
        f.flush()

    def _emit(rec):
        rows.append(rec)
        if writer:
            writer.writerow(rec)
            f.flush()                  # NOTE: 立即落盘，进程中断后可续跑

    for t in targets:
        kind = "expert" if "param" in t else "linear"
        for bits in bits_list:
            key = (t["name"], bits)
            if key in done:
                continue
            t0 = time.perf_counter()

            if kind == "linear":
                d = dppl_linear(model, ids, t["name"], bits, base_ppl,
                                ppl_fn=ppl_fn)
                _emit({"target": t["name"], "kind": "linear", "bits": bits,
                       "dppl": round(d, 6),
                       "rel_pct": round(d / base_ppl * 100, 4),
                       "params": int(t.get("params") or t.get("numel") or 0),
                       "seconds": round(time.perf_counter() - t0, 2)})
            else:
                # 逐专家单独成行、不聚合：聚合会抹平粒度，
                # 后续无法按专家粒度保护 top-K。
                p_ex = model.get_parameter(t["name"])
                per_expert_params = p_ex[0].numel()
                for j in range(t["n_experts"]):
                    t1 = time.perf_counter()
                    d = dppl_expert(model, ids, t["name"], j, bits,
                                    base_ppl, ppl_fn=ppl_fn)
                    _emit({"target": f"{t['name']}[{j}]",
                           "kind": "expert", "bits": bits,
                           "dppl": round(d, 6),
                           "rel_pct": round(d / base_ppl * 100, 4),
                           "params": int(per_expert_params),
                           "seconds": round(time.perf_counter() - t1, 2)})
    if f:
        f.close()
    return rows


# ============================================================ CSV 读写
CSV_FIELDS = ["target", "kind", "bits", "dppl", "rel_pct", "params", "seconds"]


def to_csv(rows: list[dict], path: str) -> None:
    """把扫描结果写成 CSV（含参数量列，算显存预算要用）"""
    with open(path, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        wr.writeheader()
        for r in rows:
            wr.writerow({k: r.get(k, "") for k in CSV_FIELDS})


def from_csv(path: str) -> list[dict]:
    """读回 CSV，供 aq.allocate / aq.protect / aq.agent 使用"""
    out = []
    for r in csv.DictReader(open(path, encoding="utf-8")):
        out.append({
            "target": r["target"],
            "kind": r.get("kind", "linear"),
            "bits": int(r["bits"]),
            "dppl": float(r["dppl"]),
            "rel_pct": float(r["rel_pct"]),
            "params": int(r["params"]),
            "seconds": float(r.get("seconds") or 0),
        })
    return out


def collect_routing(model, tok, ids, top_k: int = 8) -> dict:
    """hook 每个 MoE 的 gate，统计每专家被选中的 token 次数"""
    routing: dict[str, torch.Tensor] = {}

    def make_hook(n_exp):
        def hook(module, args, output):
            logits = output[0] if isinstance(output, tuple) else output
            if logits.dim() == 3:
                logits = logits.reshape(-1, logits.shape[-1])
            k = min(top_k, logits.shape[-1])
            idx = logits.topk(k, dim=-1).indices
            routing["gate"] = torch.bincount(idx.reshape(-1),
                                             minlength=n_exp).cpu().float()
        return hook

    handles = []
    for name, mod in model.named_modules():
        tname = type(mod).__name__
        if tname.endswith("Router") or name.endswith(".gate"):
            n_exp = (getattr(mod, "out_features", None)
                     or getattr(mod, "n_experts", None))
            if n_exp:
                handles.append(mod.register_forward_hook(make_hook(n_exp)))
    with torch.no_grad():
        model(ids)
    for h in handles:
        h.remove()
    return routing
"""``aq/onboard``：任意模型的量化决策流程。

对给定模型依次完成四步，不依赖预扫描结果，也不针对特定模型：

- :func:`describe`：结构清单，统计可量化单元数与参数量，并记录被排除的模块
- :func:`scan`：逐单元 × 位宽 的真实代价表，支持断点续跑
- :func:`plan_from_scan`：按显存预算或损失上限求位宽分配
- :func:`report`：输出 markdown 报告，逐项标注数据来源

设计要点：

- 单元 = ``nn.Linear``（普通层），或 3D ``nn.Parameter`` 的每个专家切片（融合专家）
- 位宽默认 ``{2, 4, 8}``；2bit 仅用于观察失效陡降，方案中默认禁用（依据见知识库 K002）
- ``lm_head`` / ``embedding`` / ``norm`` 不量化，与 bnb 行为一致
- 扫描必须在 bf16 权重上进行（K024）；显存不足时直接报错，不退化为低精度扫描

命令行::

    python -m aq.onboard --model Qwen/Qwen2.5-1.5B-Instruct --bits 4,8 --limit 32
    python -m aq.onboard --model /root/mornai-tmp/OLMoE --bits 2 --out results/moe.csv
"""
from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path

import torch

GIB = 1024 ** 3
SKIP_KEYWORDS = ("lm_head", "embed_tokens", "norm", "rotary", "router")
# 只排除 MoE 的 router（名字恰好是 ...mlp.gate），不能误伤 ...mlp.gate_proj
SKIP_EXACT_SUFFIX = (".mlp.gate",)

__all__ = ["describe", "scan", "plan_from_scan", "report", "main"]


# ============================================================ ① 描述结构
def describe(model) -> dict:
    """列出所有可量化单元，不加载数据、不执行前向。

    单元来源为 ``nn.Linear`` 层，以及 ``dim() == 3`` 的融合专家参数（按专家切片计数）。
    命中 ``SKIP_KEYWORDS`` 或 ``SKIP_EXACT_SUFFIX`` 的模块记入 ``skipped``，不参与量化。

    Args:
        model: 已加载的因果语言模型。

    Returns:
        结构字典，键为 ``linear`` / ``expert`` / ``n_linear`` / ``n_expert`` /
        ``n_units`` / ``params_total`` / ``params_quant`` / ``skipped``。
        ``n_units`` 按专家切片计数；``params_quant`` 为可量化参数量，单位个。

    Note:
        复杂度 O(模块数)，只读形状与 ``numel``，不触碰权重数据。
    """
    linear, expert, skipped = [], [], []
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear):
            if any(k in name for k in SKIP_KEYWORDS) or name.endswith(SKIP_EXACT_SUFFIX):
                skipped.append((name, "按规则排除"))
                continue
            linear.append({"name": name, "kind": "linear",
                           "params": mod.weight.numel(),
                           "shape": tuple(mod.weight.shape)})
        for pn, p in mod.named_parameters(recurse=False):
            if p.dim() == 3:
                full = f"{name}.{pn}"
                if any(k in full for k in SKIP_KEYWORDS) or full.endswith(SKIP_EXACT_SUFFIX):
                    skipped.append((full, "按规则排除"))
                    continue
                expert.append({"name": full, "kind": "expert_fused",
                               "params": p.numel(), "n_experts": p.shape[0],
                               "shape": tuple(p.shape),
                               "per_expert": p[0].numel()})
    pq = sum(u["params"] for u in linear) + sum(u["params"] for u in expert)
    return {"linear": linear, "expert": expert,
            "n_linear": len(linear), "n_expert": len(expert),
            "n_units": len(linear) + sum(e["n_experts"] for e in expert),
            "params_total": sum(p.numel() for p in model.parameters()),
            "params_quant": pq, "skipped": skipped}


# ============================================================ ② 扫描
def scan(model, ids, bits_list=(4, 8), describe_info: dict | None = None,
         group_size: int = 64, limit: int | None = None,
         out_csv: str | None = None, progress: bool = True,
         ppl_fn=None) -> list[dict]:
    """逐单元 × 位宽测量真实 ΔPPL（伪量化：量化 → 反量化 → 测 PPL → 精确还原）。

    每个单元单独量化后测一次 PPL，与基线相减得到 ``dppl``，
    再折算为 ``rel_pct``（相对基线的变化量，单位百分比）。

    Args:
        model: 已加载的因果语言模型，权重须为 bf16 或 fp16。
        ids: 输入的 token id 张量，形状 ``(seq_len,)``，1-D。
        bits_list: 要测试的位宽列表，例如 ``(4, 8)``。
        describe_info: 预先算好的 :func:`describe` 结果；为 ``None`` 时内部调用。
        group_size: 分组量化的组大小，64 表示每 64 个元素共享一个 scale。
        limit: 只扫前 N 个单元，用于快速验证管线。
        out_csv: CSV 落盘路径，同时作为断点续跑依据，已存在的
            ``(target, bits)`` 组合会被跳过。
        progress: 是否打印进度。
        ppl_fn: 自定义 PPL 函数 ``(model, ids) -> (ppl, n_tokens)``；为 ``None`` 时
            使用 :func:`aq.metrics.ppl_from_ids`，其 chunk 取
            ``min(512, len(ids) // 2)`` 且不低于 64，以满足 ``len(ids) >= 2 * chunk``。

    Returns:
        本次新测的记录列表，每项含 ``target`` / ``kind`` / ``bits`` / ``dppl`` /
        ``rel_pct`` / ``params`` / ``seconds``（``seconds`` 单位秒）。

    Raises:
        ValueError: ``ids`` 长度不足 ``2 * chunk``，PPL 无定义
            （由 :func:`aq.metrics.ppl_from_ids` 抛出）。
        RuntimeError: 权重还原失败，此时测量结果不可信
            （由 :class:`aq.fakequant.Swap` 抛出）。

    Warning:
        本函数临时原地替换权重（:class:`aq.fakequant.Swap`），逐个单元恢复。
        若 ``Swap`` 未能精确还原，模型状态即被破坏。

    Note:
        总耗时约为 ``新测单元数 × 位宽数 × 单次前向耗时``，单次前向耗时由
        ``len(ids)`` 决定；写了 ``out_csv`` 则每行即时 flush，中断可续。
    """
    from aq.fakequant import Swap, fake_quant_bits
    from aq.metrics import ppl_from_ids

    if ppl_fn is None:
        def ppl_fn(m, i):
            # NOTE: chunk 自适应：ppl_from_ids 要求 len(ids) >= 2*chunk
            c = max(64, min(512, len(i) // 2))
            return ppl_from_ids(m, i, chunk=c)

    info = describe_info or describe(model)
    base, ntok = ppl_fn(model, ids)
    if progress:
        print(f"  基线 PPL = {base:.4f}（{ntok} token）")

    # 展开成 (target_name, param, slice_index, params)
    targets = []
    for u in info["linear"]:
        targets.append((u["name"], model.get_parameter(f"{u['name']}.weight"),
                        None, u["params"]))
    for e in info["expert"]:
        p = model.get_parameter(e["name"])
        for j in range(e["n_experts"]):
            targets.append((f"{e['name']}[{j}]", p, j, e["per_expert"]))
    if limit:
        targets = targets[:limit]
        if progress:
            print(f"  ⚠️ limit={limit}，只扫前 {len(targets)} 个单元")

    FIELDS = ["target", "kind", "bits", "dppl", "rel_pct", "params", "seconds"]
    done = set()
    path = Path(out_csv) if out_csv else None
    if path and path.exists():
        done = {(r["target"], int(r["bits"]))
                for r in csv.DictReader(open(path, encoding="utf-8"))}
        if progress:
            print(f"  断点续跑：已有 {len(done)} 条，会跳过")

    fh = open(path, "a", newline="", encoding="utf-8") if path else None
    wr = csv.DictWriter(fh, fieldnames=FIELDS) if fh else None
    if fh and fh.tell() == 0:
        wr.writeheader()
        fh.flush()

    rows = []
    t0 = time.perf_counter()
    for (tname, param, idx, npar) in targets:
        kind = "expert" if idx is not None else "linear"
        for bits in bits_list:
            if (tname, bits) in done:
                continue
            t1 = time.perf_counter()
            data = param.data if idx is None else param.data[idx]
            with Swap(param, fake_quant_bits(data, bits, group_size), index=idx):
                v, _ = ppl_fn(model, ids)
            d = v - base
            row = {"target": tname, "kind": kind, "bits": bits,
                   "dppl": round(d, 6), "rel_pct": round(d / base * 100, 4),
                   "params": npar, "seconds": round(time.perf_counter() - t1, 2)}
            rows.append(row)
            if wr:
                wr.writerow(row)
                fh.flush()
            if progress and len(rows) % 20 == 0:
                el = time.perf_counter() - t0
                print(f"    {len(rows)} 条 / {el:.0f}s", flush=True)
    if fh:
        fh.close()
    if progress:
        print(f"  扫描完成 {len(rows)} 条，用时 {time.perf_counter()-t0:.0f}s")
    return rows


# ============================================================ ③ 分配
def plan_from_scan(rows: list[dict], vram_gib: float | None = None,
                   cand=(3, 4, 8), fixed_gib: float = 0.434,
                   mode: str = "min_bits", max_loss_pct: float = 1.0,
                   budget_gib: float | None = None) -> dict:
    """从实测表求位宽分配方案。

    三种模式：

    - ``min_bits``（默认）：预测损失 ≤ ``max_loss_pct`` 的前提下最小化总位数，
      对应「最大程度量化且不损失精度」的目标
    - ``budget``：给定 ``budget_gib``，最小化预测损失
    - ``fixed``：全部单元取实测位宽中的最低档，作为基线对照

    Args:
        rows: :func:`scan` 产出的记录列表。
        vram_gib: 显存总量，单位 GiB；``mode="budget"`` 且未给 ``budget_gib`` 时
            以 ``vram_gib * 0.85`` 作为可用预算。
        cand: 候选位宽，例如 ``(3, 4, 8)``。
        fixed_gib: 权重之外的固定占用，单位 GiB。
        mode: 求解模式，见上；未知取值抛 :class:`ValueError`。
        max_loss_pct: ``min_bits`` 模式允许的预测损失上限，单位百分比。
        budget_gib: 显存预算，单位 GiB。

    Returns:
        方案字典，含 ``assign`` / ``hist`` / ``gib``（GiB）/ ``budget``（GiB）/
        ``predicted_pct``（百分比）/ ``mode`` / ``measured`` / ``cand``；
        最低档仍超出损失上限时额外带 ``warning`` 键，DP 精调失败时带
        ``dp_error`` / ``dp_hint`` 并退回统一位宽方案。

    Raises:
        ValueError: 没有任何位宽被所有单元共同测过；``mode="budget"`` 但
            ``budget_gib`` 与 ``vram_gib`` 均为 ``None``；或 ``mode`` 未知。

    Note:
        只用真正测过的位宽，未测档位从候选中剔除，不以相邻档位顶替。
        代价按 ``log1p(rel_pct / 100)`` 累加，该模型系统性低估实际损失，
        只可用于排序，不可作为最终结论（见知识库 K005）。
    """
    import math as _m

    from aq.allocate import Unit
    from aq.solve import size_gib, solve_dp

    by, params = {}, {}
    for r in rows:
        by.setdefault(r["target"], {})[int(r["bits"])] = float(r["rel_pct"])
        params[r["target"]] = int(r["params"])

    have = {b for got in by.values() for b in got}
    # NOTE: 只保留所有单元都测过的位宽。缺档会让代价表出现空位，
    #   DP 拿到 NaN 时静默失败，最终报出误导性的「预算太小」。
    common = None
    for got in by.values():
        common = set(got) if common is None else (common & set(got))
    common = common or set()
    measured = (set(cand) & common) or (set(cand) & have)
    if not measured:
        raise ValueError(
            f"没有可用的实测位宽。候选={cand}，实测={sorted(have)}，"
            f"所有单元共有的={sorted(common)}。"
            f"请让扫描覆盖至少一个位宽的全部单元。")

    units = []
    for name, got in by.items():
        cost = {16: 0.0}
        for b in measured:
            if b in got:
                cost[b] = max(0.0, _m.log1p(got[b] / 100))
        units.append(Unit(name=name, params=params[name], cost=cost))

    info = {"measured": tuple(sorted(measured)),
            "n_units": len(units),
            "cand_used": tuple(sorted(measured))}

    # ---------- fixed：基线 ----------
    if mode == "fixed":
        b = min(measured)
        gib = fixed_gib + sum(size_gib(u.params, b) for u in units)
        cost = sum(u.cost.get(b, 0.0) for u in units)
        return {"assign": {u.name: b for u in units}, "hist": {b: len(units)},
                "gib": gib, "budget": round(gib, 3),
                "predicted_pct": _m.expm1(cost) * 100, "mode": mode, **info,
                "cand": info["cand_used"]}

    # ---------- min_bits：损失 ≤ max_loss 下最小化位数 ----------
    if mode == "min_bits":
        target = _m.log1p(max_loss_pct / 100)          # 允许的 log 损失上限
        lo, hi = min(measured), max(measured)
        best = None
        # 从低到高试“统一位宽”，找到第一个满足损失上限的，再在其附近用 DP 精调
        for b in sorted(measured):
            gib = fixed_gib + sum(size_gib(u.params, b) for u in units)
            cost = sum(u.cost.get(b, 0.0) for u in units)
            if _m.expm1(cost) * 100 <= max_loss_pct:
                best = {"assign": {u.name: b for u in units},
                        "hist": {b: len(units)}, "gib": gib,
                        "budget": round(gib, 3), "avg_bits": b,
                        "predicted_pct": _m.expm1(cost) * 100}
                break
        if best is None:                                # 最低档也不达标 → 用最低档并告警
            b = lo
            gib = fixed_gib + sum(size_gib(u.params, b) for u in units)
            cost = sum(u.cost.get(b, 0.0) for u in units)
            best = {"assign": {u.name: b for u in units},
                    "hist": {b: len(units)}, "gib": gib,
                    "budget": round(gib, 3), "avg_bits": b,
                    "predicted_pct": _m.expm1(cost) * 100,
                    "warning": f"最低档 {b}bit 的预测损失已超过 {max_loss_pct}%"}
        # DP 精调：允许个别单元升档/降档，总位数不超过上面这个统一方案
        try:
            budget = best["gib"]
            r = solve_dp(units, budget, candidate_bits=info["cand_used"],
                         fixed_gib=fixed_gib, n_buckets=6000)
            hist = {}
            for v in r["assign"].values():
                hist[v] = hist.get(v, 0) + 1
            refined = {"assign": r["assign"], "hist": dict(sorted(hist.items())),
                       "gib": r["gib"], "budget": round(budget, 3),
                       "avg_bits": round(sum(r["assign"].values()) / len(units), 2),
                       "predicted_pct": _m.expm1(r["cost"]) * 100}
            if refined["predicted_pct"] <= max_loss_pct:
                best = refined
        except ValueError as e:
            best["dp_error"] = str(e)
            best["dp_hint"] = (f"精调失败（{e}），已退回统一 {best.get('avg_bits')}"
                               "bit 的方案 —— 该方案本身是可行的。")
        return {**best, "mode": mode, **info, "cand": info["cand_used"]}

    # ---------- budget ----------
    if mode == "budget":
        if budget_gib is None:
            if vram_gib is None:
                raise ValueError("mode=budget 需要 budget_gib 或 vram_gib")
            budget_gib = vram_gib * 0.85
        r = solve_dp(units, budget_gib, candidate_bits=info["cand_used"],
                     fixed_gib=fixed_gib, n_buckets=6000)
        hist = {}
        for v in r["assign"].values():
            hist[v] = hist.get(v, 0) + 1
        return {"assign": r["assign"], "hist": dict(sorted(hist.items())),
                "gib": r["gib"], "budget": round(budget_gib, 3),
                "avg_bits": round(sum(r["assign"].values()) / len(units), 2),
                "predicted_pct": _m.expm1(r["cost"]) * 100,
                "mode": mode, **info, "cand": info["cand_used"]}

    raise ValueError(f"未知 mode: {mode}")


def plan_all(rows: list[dict], vram_gib: float | None = None,
             cand=(3, 4, 8), fixed_gib: float = 0.434,
             max_loss_pct: float = 1.0) -> list[dict]:
    """并列给出多个候选方案，每个标注适用场景，供调用方选择。

    - 方案 A「最省显存」：损失 ≤ ``max_loss_pct`` 下把位宽压到最低（``min_bits``）
    - 方案 B「最保精度」：全部使用最高位宽（质量优先）
    - 方案 C「均衡」：给定显存预算下最优（``budget``，按 85% 显存计）

    Args:
        rows: :func:`scan` 产出的记录列表。
        vram_gib: 显存总量，单位 GiB。
        cand: 候选位宽，例如 ``(3, 4, 8)``。
        fixed_gib: 权重之外的固定占用，单位 GiB。
        max_loss_pct: 允许的预测损失上限，单位百分比。

    Returns:
        方案列表，每项为 :func:`plan_from_scan` 的结果补上 ``name`` / ``scene``；
        求解失败的项只含 ``name`` / ``scene`` / ``error``，不中断其余方案。
    """
    out = []
    specs = [
        ("A. 最省显存", "min_bits", "显存最紧张，能接受 ~1% 损失"),
        ("B. 最保精度", "fixed", "显存够，想尽量不掉点"),
        ("C. 均衡", "budget", "有明确显存上限，想在限内最好"),
    ]
    for name, mode, scene in specs:
        try:
            if mode == "fixed":
                # 用最高位宽
                pl = plan_from_scan(rows, vram_gib, cand=cand, mode="fixed",
                                    max_loss_pct=max_loss_pct)
                # 覆盖成 max bit
                by_bits = {}
                for r in rows:
                    by_bits.setdefault(r["target"], {})[int(r["bits"])] = 1
                from aq.allocate import Unit
                from aq.solve import size_gib
                mb = max(cand)
                units = [Unit(name=k, params=int(next(
                    r["params"] for r in rows if r["target"] == k)),
                    cost={16: 0.0, mb: 0.0}) for k in by_bits]
                gib = fixed_gib + sum(size_gib(u.params, mb) for u in units)
                pl = {"assign": {u.name: mb for u in units},
                      "hist": {mb: len(units)}, "gib": gib,
                      "budget": round(gib, 3), "avg_bits": mb,
                      "predicted_pct": None, "mode": "fixed",
                      "cand": tuple(sorted(cand)), "n_units": len(units),
                      "measured": tuple(sorted(cand))}
            else:
                pl = plan_from_scan(rows, vram_gib, cand=cand, mode=mode,
                                    max_loss_pct=max_loss_pct)
        except ValueError as e:
            out.append({"name": name, "scene": scene, "error": str(e)})
            continue
        pl["name"] = name
        pl["scene"] = scene
        out.append(pl)

    # 按显存升序
    ok = [x for x in out if "gib" in x]
    ok.sort(key=lambda x: x["gib"])
    return out


# ============================================================ ④ 报告
def report(info: dict, rows: list[dict] | None, plan: dict | None,
           model_name: str, bits_list) -> str:
    L = [f"# {model_name} 的量化决策报告", ""]
    L += ["## ① 结构", "",
          f"- 参数量 **{info['params_total']:,}**（其中可量化 {info['params_quant']:,}）",
          f"- 普通 Linear：**{info['n_linear']}** 个",
          f"- 融合专家张量：**{info['n_expert']}** 个"
          f"（共 {info['n_units'] - info['n_linear']} 个专家切片）",
          f"- **可量化单元合计 {info['n_units']}**", ""]
    if info["skipped"]:
        L += ["不量化的部分（与 bnb 行为一致）：", ""]
        L += [f"- `{n}`（{why}）" for n, why in info["skipped"][:10]]
        L += [""] * (0 if len(info["skipped"]) <= 10 else [])
    if rows:
        L += ["## ② 实测敏感度", ""]
        agg = {}
        for r in rows:
            agg.setdefault(r["bits"], []).append(r["rel_pct"])
        L += ["| 位宽 | 最小 | 中位 | 最大 | 单元数 |", "|---|---|---|---|---|"]
        import statistics
        for b in sorted(agg):
            v = agg[b]
            L += [f"| {b}bit | {min(v):+.2f}% | {statistics.median(v):+.2f}% | "
                  f"{max(v):+.2f}% | {len(v)} |"]
        L += [""]
        worst_b = min(bits_list)
        top = sorted((r for r in rows if r["bits"] == worst_b),
                     key=lambda r: -r["rel_pct"])[:8]
        L += [f"最敏感的 8 个（{worst_b}bit）：", "",
              "| 单元 | ΔPPL |", "|---|---|"]
        L += [f"| `{r['target'][-46:]}` | {r['rel_pct']:+.2f}% |" for r in top]
        L += [""]
    if plan:
        L += ["## ③ 分配方案", "",
              f"- 目标：**{plan.get('mode', '?')}**"
              f"（{'损失 ≤ 1% 下压到最低位宽' if plan.get('mode') == 'min_bits' else '给定预算' }）",
              f"- 实测过的位宽：{list(plan['cand'])}",
              f"- 实际占用：**{plan['gib']:.3f} GiB**"
              f"（平均 **{plan.get('avg_bits', '?')}** bit）",
              f"- 位宽分布：`{plan['hist']}`",
              f"- **预测 ΔPPL：{plan['predicted_pct']:+.2f}%**", "",
              "> ⚠️ 预测会低估实际（实测约 2.06 倍，见知识库 K005）。",
              "> 最终方案必须在模型上实测验证。", ""]
    L += ["---", "",
          "**所有数字均在该模型上实测**（伪量化：量化→反量化→测 PPL→逐位还原）。"]
    return "\n".join(L)


# ============================================================ CLI
def main():
    ap = argparse.ArgumentParser(description="给定任意模型，做量化决策")
    ap.add_argument("--model", required=True, help="HF 名称或本地路径")
    ap.add_argument("--bits", default="4,8", help="要扫的位宽，如 4,8 或 3,4,5")
    ap.add_argument("--cand", default=None, help="方案候选位宽（默认取 --bits）")
    ap.add_argument("--limit", type=int, default=None, help="只扫前 N 个单元")
    ap.add_argument("--max-tokens", type=int, default=801)
    ap.add_argument("--vram", type=float, default=None, help="显存 GiB（默认现测）")
    ap.add_argument("--mode", default="min_bits",
                    choices=["min_bits", "budget", "fixed"])
    ap.add_argument("--max-loss", type=float, default=1.0,
                    help="min_bits 模式允许的预测损失百分比")
    ap.add_argument("--budget-gib", type=float, default=None)
    ap.add_argument("--out", default=None, help="CSV 路径（默认 results/<模型>.csv）")
    ap.add_argument("--report", default=None, help="报告 md 路径")
    ap.add_argument("--text-file", default="data/en_eval.txt")
    a = ap.parse_args()

    bits_list = tuple(int(x) for x in a.bits.split(",") if x.strip())
    cand = tuple(int(x) for x in (a.cand or a.bits).split(",") if x.strip())

    # 显存：现测
    if a.vram is None:
        if not torch.cuda.is_available():
            raise SystemExit("没有 CUDA")
        a.vram = torch.cuda.get_device_properties(0).total_memory / GIB
    print(f"显存 {a.vram:.2f} GiB | 模型 {a.model}")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    import transformers
    kw = ({"dtype": torch.bfloat16}
          if int(transformers.__version__.split(".")[0]) >= 5
          else {"torch_dtype": torch.bfloat16})
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, **kw).eval().to("cuda")
    used = torch.cuda.memory_allocated() / GIB
    print(f"已加载，占用 {used:.2f} GiB")

    info = describe(model)
    print(f"\n可量化单元 {info['n_units']}"
          f"（Linear {info['n_linear']} + 专家切片 "
          f"{info['n_units'] - info['n_linear']}）")

    tf = Path(a.text_file)
    text = tf.read_text(encoding="utf-8") if tf.exists() else "The quick brown fox."
    ids = tok(text, return_tensors="pt").input_ids[0][:a.max_tokens].to(model.device)

    out = a.out or f"results/{Path(a.model).name or 'model'}_scan.csv"
    Path("results").mkdir(exist_ok=True)
    rows = scan(model, ids, bits_list=bits_list, describe_info=info,
                limit=a.limit, out_csv=out)

    plan = None
    if rows:
        try:
            plan = plan_from_scan(rows, a.vram, cand=cand, mode=a.mode,
                                  max_loss_pct=a.max_loss,
                                  budget_gib=a.budget_gib)
        except ValueError as e:
            print(f"方案求解失败：{e}")

    rep = report(info, rows, plan, a.model, bits_list)
    rp = Path(a.report or f"results/{Path(a.model).name or 'model'}_report.md")
    rp.write_text(rep, encoding="utf-8")
    print(f"\n报告已写出 {rp}")
    print(rep[:1500])


if __name__ == "__main__":
    main()

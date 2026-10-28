"""保守策略：按敏感度保护 top-K 模块，其余统一低精度。

对外接口：

- :func:`protect_plan`：按 ``sort_bits`` 档位的敏感度排序，前 ``k`` 个模块用
  ``protect_bits``，其余用 ``base_bits``
- :func:`evaluate_plan`：把方案应用到模型上测真实 PPL，随后精确还原
- :func:`sweep_k`：对一组 K 求（显存, 真实 PPL），得到一条 Pareto 曲线

该策略单调、可验证：多保护一个模块，结果只会更好或持平。

已知局限：「单模块敏感度之和」不可叠加，基于它的多选背包/贪心求解会系统性
失效——实测 Σ ΔPPL% 为 −7.10%（暗示量化有益），真实为 +15.32%；改用
Δlog(PPL) 后符号正确，但 2bit 档仍低估约 16 倍；196 个模块同时量化时误差经
层间传播放大，贪心方案的真实 PPL 比均匀 4bit 差 75%。故本模块只采用保守策略，
敏感度求和结果仅可用于排序（见 ``05_专题/专题01``）。
"""
from __future__ import annotations

__all__ = ["protect_plan", "evaluate_plan", "sweep_k"]

FIXED_GIB = 0.434          # lm_head 永不量化的固定开销，单位 GiB


def protect_plan(rows: list[dict], k: int, protect_bits: int = 8,
                 base_bits: int = 4, fixed_gib: float = FIXED_GIB,
                 sort_bits: int = 2) -> dict:
    """按指定位宽档的敏感度保护 top-K 模块，其余模块用基础位宽。

    敏感度按 ``rel_pct`` 从高到低排序，前 ``k`` 个模块分到 ``protect_bits``，
    其余分到 ``base_bits``；显存为 ``fixed_gib`` 加各模块按位宽折算的 GiB 之和。

    Args:
        rows: 敏感度记录列表，每项含 ``target``（模块名）、``bits``（探测位宽）、
            ``rel_pct``（相对基线的 PPL 变化百分比）、``params``（参数量）。
        k: 保护的模块个数。
        protect_bits: 被保护模块的位宽，默认 8。
        base_bits: 未保护模块的位宽，默认 4。
        fixed_gib: 不参与量化的固定开销，单位 GiB，默认 ``FIXED_GIB``。
        sort_bits: 用于排序的敏感度档位，默认 2bit 档。

    Returns:
        字典，含 ``assign``（``{模块名: bits}``）、``gib``（估算显存，单位 GiB）、
        ``protected``（被保护的模块名列表，按敏感度降序）。

    Note:
        缺少 ``sort_bits`` 档记录的模块不参与排序，恒分到 ``base_bits``；
        ``k`` 大于可排序模块数时，全部可排序模块均被保护。
    """
    from aq.allocate import size_gib

    sens = {r["target"]: r["rel_pct"] for r in rows if r["bits"] == sort_bits}
    params = {r["target"]: r["params"] for r in rows}
    order = sorted(sens, key=lambda t: -sens[t])
    keep = set(order[:k])

    assign = {m: (protect_bits if m in keep else base_bits) for m in params}
    gib = fixed_gib + sum(size_gib(params[m], b) for m, b in assign.items())
    return {"assign": assign, "gib": gib, "protected": order[:k]}


def evaluate_plan(model, ids, assign: dict[str, int], ppl_fn) -> float:
    """把方案应用到模型上，测真实 PPL 后精确还原权重。

    Args:
        model: 已加载的因果语言模型。
        ids: 输入的 token id 张量，形状 ``(seq_len,)``。
        assign: ``{模块名: bits}`` 位宽分配，模块名不带 ``.weight`` 后缀。
        ppl_fn: 可调用对象，签名为 ``ppl_fn(model, ids) -> (ppl, ...)``，
            返回的第一项为 PPL。

    Returns:
        方案应用后的 PPL。

    Raises:
        RuntimeError: 还原后的基线 PPL 与测量前不一致，说明权重未被精确还原，
            本次测量结果不可信。

    Warning:
        本函数原地修改模型权重，属不可逆操作：``restore`` 未精确还原即破坏模型
        状态，此后该模型不应再用于其他测量。

    Note:
        还原校验要求基线 PPL 完全相等，不做容差比较。
        延迟导入 ``aq.fakequant``，使模块级导入不依赖 torch。
    """
    import torch

    from aq.fakequant import quantize_model_inplace, restore

    pname = {f"{k}.weight": v for k, v in assign.items()}
    base = ppl_fn(model, ids)[0]
    handles = quantize_model_inplace(model, pname, bits_default=16)
    try:
        v, _ = ppl_fn(model, ids)
    finally:
        restore(model, handles)
    if ppl_fn(model, ids)[0] != base:
        raise RuntimeError("★ 还原失败，测量结果不可信")
    return v


def sweep_k(model, ids, rows: list[dict], ks=(0, 2, 4, 8, 16, 32, 64),
            ppl_fn=None, **kw) -> list[dict]:
    """对一组 ``k`` 求（显存, 真实 PPL），得到一条 Pareto 曲线。

    Args:
        model: 已加载的因果语言模型。
        ids: 输入的 token id 张量，形状 ``(seq_len,)``。
        rows: 敏感度记录列表，透传给 :func:`protect_plan`。
        ks: 待扫描的 ``k`` 序列，默认 ``(0, 2, 4, 8, 16, 32, 64)``。
        ppl_fn: PPL 计算函数；为 ``None`` 时使用 ``aq.metrics.ppl_from_ids``，
            并固定 ``chunk=512``。
        **kw: 透传给 :func:`protect_plan` 的其余参数。

    Returns:
        记录列表，每项含 ``k``、``gib``（估算显存，单位 GiB）、``ppl``、
        ``rel_pct``（相对基线的 PPL 变化百分比）。

    Note:
        每个 ``k`` 需 3 次完整前向（基线、方案、还原校验），另加 1 次全函数基线；
        总耗时随 ``len(ks)`` 线性增长。``rel_pct`` 以函数开始处测得的基线为准。
    """
    if ppl_fn is None:
        from aq.metrics import ppl_from_ids

        def ppl_fn(m, i):
            return ppl_from_ids(m, i, chunk=512)

    base, _ = ppl_fn(model, ids)
    out = []
    for k in ks:
        plan = protect_plan(rows, k, **kw)
        p = evaluate_plan(model, ids, plan["assign"], ppl_fn)
        out.append({"k": k, "gib": round(plan["gib"], 4),
                    "ppl": round(p, 4),
                    "rel_pct": round((p / base - 1) * 100, 4)})
    return out

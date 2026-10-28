"""位宽分配求解器：在给定显存预算下为每层 / 每个专家选择位宽。

以多选背包形式建模：

- 变量：单元 ``i`` 的位宽 ``b_i``，取值 ``{2, 4, 8, 16}``
- 目标：最小化 Σ ΔPPL_i(b_i)
- 约束：Σ size_i(b_i) ≤ 预算 B

``Δ_i(b)`` 随 ``b`` 递减且边际收益递减，贪心的 1/2 近似在此适用。
实测可加性为 1.02–1.10×，故「Σ 单单元 ΔPPL」可作为可信的排序目标函数。

尺寸按有效位宽计算，而非「参数量 × bits/8」：4bit 的 scale 开销使实测值
达到 4.128 bits/参数（含 double quantization）。
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

GIB = 1024 ** 3
DEFAULT_BITS = (2, 4, 8, 16)          # 候选位宽（16 = 不量化）
BITS_OVERHEAD = {4: 4.128, 8: 8.0, 2: 2.0, 16: 16.0}


# ============================================================ 尺寸函数
def size_gib(params: int, bits: int) -> float:
    """计算单个单元的权重显存。

    Args:
        params: 单元参数量。
        bits: 目标位宽，即 ``BITS_OVERHEAD`` 的键。

    Returns:
        显存占用，单位 GiB。

    Note:
        4bit 按有效位宽 4.128 bits/参数 计算，其中含 scale 开销。
    """
    eff = BITS_OVERHEAD.get(bits, float(bits))
    return params * eff / 8 / GIB


def total_gib(assign: dict[str, int], params: dict[str, int],
              fixed_gib: float = 0.0) -> float:
    """计算一个完整方案的显存。

    Args:
        assign: 单元名到位宽的映射。
        params: 单元名到参数量的映射，须覆盖 ``assign`` 的全部键。
        fixed_gib: 未量化部分的固定开销，单位 GiB，例如 lm_head 的 0.434 GiB。

    Returns:
        总显存，单位 GiB。
    """
    return fixed_gib + sum(size_gib(params[k], b) for k, b in assign.items())


# ============================================================ 数据结构
@dataclass
class Unit:
    """一个可分配的单元，对应一层或一个专家张量。

    Attributes:
        name: 单元标识，与 ``rows`` 中的 ``target`` 一致。
        params: 该单元的参数量。
        cost: 位宽到 ΔPPL 百分比的映射。
    """
    name: str
    params: int
    cost: dict[int, float] = field(default_factory=dict)   # bits -> ΔPPL

    def delta(self, bits: int) -> float:
        return self.cost.get(bits, 0.0)


def units_from_csv(rows: list[dict], bits_all: tuple[int, ...] = DEFAULT_BITS
                   ) -> list[Unit]:
    """从 ``aq.sensitivity`` 的 ``rows`` 构造 ``Unit`` 列表。

    Args:
        rows: 每项含 ``target`` / ``bits`` / ``rel_pct`` / ``params``。
        bits_all: 需要覆盖的位宽档位，默认为 ``DEFAULT_BITS``。

    Returns:
        与 ``rows`` 中 ``target`` 一一对应的 ``Unit`` 列表。

    Note:
        缺失的位宽档位按最近的已知档位近似，不能填 0：
        填 0 会使贪心误判该档位无损失，等价于免费升级。
    """
    by: dict[str, dict[int, float]] = {}
    params: dict[str, int] = {}
    for r in rows:
        by.setdefault(r["target"], {})[int(r["bits"])] = float(r["rel_pct"])
        params[r["target"]] = int(r["params"])

    out = []
    for name, got in by.items():
        cost = {}
        for b in bits_all:
            if b in got:
                cost[b] = got[b]
            elif b == 16:
                cost[b] = 0.0                     # 不量化 = 无损失
            else:
                # 用最近的已知档位近似
                near = min(got, key=lambda x: abs(x - b))
                cost[b] = got[near]
        out.append(Unit(name=name, params=params[name], cost=cost))
    return out


# ============================================================ 贪心分配
def greedy_allocate(units: list[Unit], budget_gib: float,
                    candidate_bits=DEFAULT_BITS,
                    fixed_gib: float = 0.0) -> dict:
    """用贪心求解多选背包，给出显存预算下的位宽方案。

    流程：

    1. 全部单元从最低位宽起算，得到已用显存；
    2. 反复选取「每多花 1 GiB 换来的 ΔPPL 降幅」最大的单步升档；
    3. 预算用尽或无可升档位时停止。

    Args:
        units: 待分配单元列表。
        budget_gib: 总显存预算，单位 GiB，含 ``fixed_gib``。
        candidate_bits: 候选位宽，默认 ``DEFAULT_BITS``。
        fixed_gib: 未量化部分的固定开销，单位 GiB。

    Returns:
        字典，含 ``assign``（单元名到位宽）、``dppl``（总 ΔPPL，百分比）、
        ``gib``（总显存，GiB）与 ``log``（每次升档的记录）。

    Raises:
        ValueError: 预算不足以容纳所有单元取最低位宽的方案。

    Note:
        复杂度 O(k·n·|bits|)，``n`` 为单元数、``k`` 为升档次数；每次升档都重扫
        全部候选，数百单元规模下单次调用为亚秒级。
    """
    bits_sorted = sorted(candidate_bits)
    lo = bits_sorted[0]
    params = {u.name: u.params for u in units}

    assign = {u.name: lo for u in units}
    used = total_gib(assign, params, fixed_gib)
    if used > budget_gib:
        raise ValueError(
            f"预算 {budget_gib:.3f} GiB 装不下最低位宽（{lo}bit）方案，"
            f"需要 {used:.3f} GiB")

    log = []
    while True:
        best = None                       # (效率, 单元, 新bits, Δgib, 收益)
        for u in units:
            b_now = assign[u.name]
            for b_next in (b for b in bits_sorted if b > b_now):
                dg = size_gib(u.params, b_next) - size_gib(u.params, b_now)
                if used + dg > budget_gib:
                    continue
                dp = u.delta(b_next) - u.delta(b_now)    # 升位宽后 ΔPPL 下降
                gain = -dp                              # 正的收益
                if gain <= 0:
                    continue
                eff = gain / dg if dg > 1e-15 else float("inf")
                if best is None or eff > best[0]:
                    best = (eff, u, b_next, dg, gain)

        if best is None:
            break
        eff, u, b_next, dg, gain = best
        old = assign[u.name]
        assign[u.name] = b_next
        used += dg
        log.append({"unit": u.name, "from": old, "to": b_next,
                    "dgib": round(dg, 6), "dppl_gain": round(gain, 6),
                    "eff": round(eff, 4)})

    dppl = sum(u.delta(assign[u.name]) for u in units)
    return {"assign": assign, "dppl": dppl, "gib": used, "log": log}


# ============================================================ 对照：均匀
def uniform_allocate(units: list[Unit], bits: int) -> dict:
    """按同一固定位宽分配所有单元，即全局 N bit 基线。

    Args:
        units: 待分配单元列表。
        bits: 统一位宽。

    Returns:
        与 :func:`greedy_allocate` 同结构的字典，``log`` 为空列表。
    """
    assign = {u.name: bits for u in units}
    return {"assign": assign,
            "dppl": sum(u.delta(bits) for u in units),
            "gib": sum(size_gib(u.params, bits) for u in units),
            "log": []}


# ============================================================ 预算扫描 → Pareto
def pareto_sweep(units: list[Unit], budgets: list[float],
                 candidate_bits=DEFAULT_BITS,
                 fixed_gib: float = 0.0) -> list[dict]:
    """对多个预算各跑一次贪心，得到 (显存, ΔPPL) 点集。

    Args:
        units: 待分配单元列表。
        budgets: 预算列表，单位 GiB。
        candidate_bits: 候选位宽，默认 ``DEFAULT_BITS``。
        fixed_gib: 未量化部分的固定开销，单位 GiB。

    Returns:
        每个可行预算对应一项结果，并附 ``budget`` 字段；预算过低、
        触发 ``ValueError`` 的预算被跳过。
    """
    out = []
    for b in budgets:
        try:
            r = greedy_allocate(units, b, candidate_bits, fixed_gib)
        except ValueError:
            continue
        r["budget"] = b
        out.append(r)
    return out


def summarize(res: dict) -> dict:
    """汇总一个方案的显存、ΔPPL 与位宽分布，供报告使用。

    Args:
        res: :func:`greedy_allocate` 或 :func:`uniform_allocate` 的返回值。

    Returns:
        含 ``gib`` / ``dppl`` / ``bits_hist`` / ``n_units`` 的字典，
        前两者保留 4 位小数。
    """
    return {
        "gib": round(res["gib"], 4),
        "dppl": round(res["dppl"], 4),
        "bits_hist": dict(sorted(Counter(res["assign"].values()).items())),
        "n_units": len(res["assign"]),
    }

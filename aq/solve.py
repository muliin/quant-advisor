"""多选背包（MCKP）位宽分配求解器。

在给定显存预算下为每个可量化单元选择一个位宽，使代价之和最小::

    min Σ_l c_l(b_l)
    s.t. Σ_l N_l·b_l ≤ B        （位宽预算，等价于显存预算）
         b_l ∈ {2, 4, 8, 16}

提供三条公开路径：

- :func:`solve_dp`：预算离散化后做动态规划，求全局最优。复杂度
  ``O(n·K·n_buckets)``，n = 单元数，K = 候选位宽数；默认 ``n_buckets=20000``、
  ``K=4``、数百个单元时状态转移约 ``1e7`` 量级，单次求解在秒级
- :func:`solve_lagrange`：拉格朗日松弛，在连续松弛下精确，且一次即可给出
  整条 Pareto 前沿
- :func:`pareto_front_mckp`：在一串预算上依次调用上述求解器并收集结果

预算与容量的单位统一为 GiB，位宽的单位为 bits。求解器只对传入的代价表求最优，
不校验代价模型本身是否准确，两者需分别评估。
"""
from __future__ import annotations

import numpy as np

GIB = 1024 ** 3
BITS_OVERHEAD = {2: 2.0, 3: 3.0, 4: 4.128, 8: 8.0, 16: 16.0}

__all__ = ["solve_dp", "solve_lagrange", "pareto_front_mckp", "size_gib"]


def size_gib(params: int, bits: int) -> float:
    """按位宽估算单个单元的权重显存占用。

    Args:
        params: 单元参数量，单位为「个」。
        bits: 位宽，单位为 bits。``BITS_OVERHEAD`` 中的档位取含 scale 开销的
            有效位宽（4bit 记为 4.128），未列出的档位按名义位宽计算。

    Returns:
        显存占用，单位 GiB。
    """
    eff = BITS_OVERHEAD.get(bits, float(bits))
    return params * eff / 8 / GIB


def _prep(units, candidate_bits):
    """把 units 整理成 (names, params[j], cost[j][k])"""
    bits = sorted(candidate_bits)
    names = [u.name for u in units]
    params = np.array([u.params for u in units], dtype=np.float64)
    cost = np.array([[float(u.cost.get(b, np.nan)) for b in bits] for u in units],
                    dtype=np.float64)
    # 缺档位用最近已知档位补，避免 NaN 让求解器以为"免费"
    for j in range(cost.shape[0]):
        col = cost[j]
        if np.all(np.isnan(col)):
            cost[j] = 0.0
            continue
        idx = np.where(~np.isnan(col))[0]
        for k in range(len(bits)):
            if np.isnan(col[k]):
                cost[j, k] = col[idx[np.argmin(np.abs(idx - k))]]
    return bits, names, params, cost


# ============================================================ DP（精确）
def solve_dp(units, budget_gib: float, candidate_bits=(2, 4, 8, 16),
             fixed_gib: float = 0.0, n_buckets: int = 20000) -> dict:
    """
    在离散化预算下求全局最优的位宽分配。

    以 ``budget_gib - fixed_gib`` 为可用预算，等分为 ``n_buckets`` 个桶后做
    多重选择背包 DP，每个桶宽 ``B / n_buckets`` GiB。

    Args:
        units: 可量化单元序列，每项须有 ``name`` / ``params`` / ``cost``，
            ``cost[b]`` 为位宽 b 下的代价。
        budget_gib: 总显存预算，单位 GiB。
        candidate_bits: 候选位宽，单位 bits。
        fixed_gib: 预算中不参与分配的部分（KV cache、激活等），单位 GiB。
        n_buckets: 预算等分桶数；桶越细离散化误差越小，复杂度随之线性增长。

    Returns:
        字典，含 ``assign``（单元名 → 位宽）、``cost``（代价之和）、``gib``
        （实际占用，GiB）、``bits_hist``（各档位单元数）、``method``。

    Raises:
        ValueError: 可用预算 ``budget_gib - fixed_gib`` 非正。
        ValueError: 预算小到无法容纳所有单元的最低档位。

    Note:
        离散化误差不超过 1 个桶，相对预算误差约 ``1 / n_buckets``。
    """
    bits, names, params, cost = _prep(units, candidate_bits)
    B = budget_gib - fixed_gib
    if B <= 0:
        raise ValueError(f"预算 {budget_gib:.4f} ≤ 固定开销 {fixed_gib:.4f}")

    w = np.array([[size_gib(p, b) for b in bits] for p in params])  # (n, K)
    unit = B / n_buckets                                     # 每桶多少 GiB
    # 用 floor 而不是 round：round 会向上取整，n 个单元累计后可能超出预算，
    # 导致「全最低档」都不可行；floor 宁可低估重量，保证可行解存在
    wi = np.maximum(1, np.floor(w / unit).astype(np.int64))
    cap = n_buckets

    INF = np.inf
    dp = np.full(cap + 1, INF)
    dp[0] = 0.0
    choice = np.zeros((len(names), cap + 1), dtype=np.int8)

    for j in range(len(names)):
        # nd 从 INF 起：每个单元都必须被分配，不能继承上一单元的未分配状态
        nd = np.full(cap + 1, INF)
        ch = np.zeros(cap + 1, dtype=np.int8)
        for k in range(len(bits)):
            wk, ck = wi[j, k], cost[j, k]
            if wk > cap:
                continue
            cand = np.full(cap + 1, INF)
            cand[wk:] = dp[:cap + 1 - wk] + ck
            upd = cand < nd
            nd[upd] = cand[upd]
            ch[upd] = k
        dp = nd
        choice[j] = ch

    # dp[c] 是「恰好用 c 个桶」的最小代价（不是 ≤ c），
    # 因此要在所有 c ≤ cap 里取代价最小的点，而不是只认 dp[cap]。
    j_point = int(np.argmin(dp))
    if not np.isfinite(dp[j_point]):
        raise ValueError("预算太小，连最低位宽方案都装不下")

    assign = {}
    cap_now = j_point
    for j in range(len(names) - 1, -1, -1):
        k = int(choice[j, cap_now])
        assign[names[j]] = bits[k]
        cap_now -= int(wi[j, k])
        if cap_now < 0:
            cap_now = 0

    used = fixed_gib + sum(size_gib(params[i], assign[names[i]])
                           for i in range(len(names)))
    return {"assign": assign, "cost": float(dp[j_point]), "gib": used,
            "bits_hist": {b: list(assign.values()).count(b) for b in bits
                          if list(assign.values()).count(b)},
            "method": f"dp({n_buckets}buckets)"}


# ============================================================ 拉格朗日
def solve_lagrange(units, budget_gib: float, candidate_bits=(2, 4, 8, 16),
                   fixed_gib: float = 0.0, iters: int = 60) -> dict:
    """
    用拉格朗日松弛求位宽分配，并修补为可行解。

    给定乘子 λ 时各单元独立取 ``argmin_k (c_k + λ·w_k)``，对 λ 二分使总占用
    逼近预算；最终仍超预算则按「降档损失 / 省下显存」升序逐个降档。

    Args:
        units: 可量化单元序列，要求同 :func:`solve_dp`。
        budget_gib: 总显存预算，单位 GiB。
        candidate_bits: 候选位宽，单位 bits。
        fixed_gib: 预算中不参与分配的部分，单位 GiB。
        iters: λ 的二分迭代次数上限。

    Returns:
        字典，含 ``assign``（单元名 → 位宽）、``cost``（代价之和）、``gib``
        （实际占用，GiB）、``lam``（最终乘子）、``method``。

    Raises:
        ValueError: 可用预算 ``budget_gib - fixed_gib`` 非正。

    Note:
        仅连续松弛下的最优性精确；位宽取整数并经降档修补后不再保证最优，
        大批量单元时以本路径做快速近似。
    """
    bits, names, params, cost = _prep(units, candidate_bits)
    w = np.array([[size_gib(p, b) for b in bits] for p in params])
    B = budget_gib - fixed_gib
    if B <= 0:
        raise ValueError("预算 ≤ 固定开销")

    lo_lam, hi_lam = 0.0, 1e6
    best = None
    for _ in range(iters):
        lam = (lo_lam + hi_lam) / 2
        idx = np.argmin(cost + lam * w, axis=1)
        g = float((w[np.arange(len(names)), idx]).sum())
        if g <= B:
            best = (lam, idx)
            hi_lam = lam
        else:
            lo_lam = lam
        if abs(g - B) < 1e-6:
            break

    if best is None:                     # 全部超预算 → 用最低档
        idx = np.zeros(len(names), dtype=int)
        lam = hi_lam
    else:
        lam, idx = best

    assign = {names[j]: bits[int(idx[j])] for j in range(len(names))}
    used = fixed_gib + float((w[np.arange(len(names)), idx]).sum())
    # 修补：若仍超预算，逐个把"降档损失/省下显存"最小的降一档
    while used > budget_gib:
        cands = []
        for j in range(len(names)):
            k = list(bits).index(assign[names[j]])
            if k == 0:
                continue
            dg = w[j, k] - w[j, k - 1]
            dloss = cost[j, k - 1] - cost[j, k]
            if dg > 0:
                cands.append((dloss / dg, j, k))
        if not cands:
            break
        cands.sort()
        _, j, k = cands[0]
        used -= float(w[j, k] - w[j, k - 1])
        assign[names[j]] = bits[k - 1]

    c = float(sum(cost[list(names).index(n), list(bits).index(b)]
                  for n, b in assign.items()))
    return {"assign": assign, "cost": c, "gib": used, "lam": lam,
            "method": "lagrange"}


# ============================================================ Pareto
def pareto_front_mckp(units, budgets, candidate_bits=(2, 4, 8, 16),
                      fixed_gib: float = 0.0, solver="dp") -> list[dict]:
    """对一串预算求解，返回前沿点（不含非支配过滤，留给 aq.pareto）。

    Args:
        units: 可量化单元序列，要求同 :func:`solve_dp`。
        budgets: 待求解的预算序列，每项单位 GiB。
        candidate_bits: 候选位宽，单位 bits。
        fixed_gib: 预算中不参与分配的部分，单位 GiB。
        solver: ``"dp"`` 走 :func:`solve_dp`，其余取值走 :func:`solve_lagrange`。

    Returns:
        字典列表，每项为对应预算的解并额外带 ``budget`` 字段；非支配过滤由
        :mod:`aq.pareto` 负责。

    Note:
        单个预算不可行时抛出的 :class:`ValueError` 被吞掉并跳过该点，
        因此返回长度可能小于 ``budgets`` 长度。
    """
    fn = solve_dp if solver == "dp" else solve_lagrange
    out = []
    for b in budgets:
        try:
            r = fn(units, b, candidate_bits, fixed_gib)
        except ValueError:
            continue
        r["budget"] = b
        out.append(r)
    return out

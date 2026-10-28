"""Pareto 前沿工具：非支配排序与结果绘图。

公开接口：

- :func:`dominates`：判断单点的支配关系
- :func:`pareto_front`：从候选点集中筛出非支配集
- :func:`to_points` / :func:`plot`：统一为 ``{gib, dppl, label}`` 记录并出图

支配关系定义：A 支配 B，当且仅当 A 在所有维度上不劣于 B，且至少一维严格更优。
本模块的两个维度均取「越小越好」：显存（单位 GiB）、ΔPPL（单位百分比）。
"""
from __future__ import annotations

__all__ = ["dominates", "pareto_front", "to_points", "plot"]


def dominates(a: dict, b: dict, keys: tuple[str, ...] = ("gib", "dppl")) -> bool:
    """判断 ``a`` 是否支配 ``b``。

    支配成立需同时满足两条：``a`` 在每个维度上不劣于 ``b``，且至少一维严格更优。

    Args:
        a: 候选点，须包含 ``keys`` 中的全部维度。
        b: 参照点，须包含 ``keys`` 中的全部维度。
        keys: 参与比较的维度名，默认 ``("gib", "dppl")``，均为越小越好。

    Returns:
        支配成立返回 ``True``，否则返回 ``False``。

    Raises:
        KeyError: ``a`` 或 ``b`` 缺少 ``keys`` 中的某个维度。
    """
    not_worse = all(a[k] <= b[k] for k in keys)
    better = any(a[k] < b[k] for k in keys)
    return not_worse and better


def pareto_front(points: list[dict], keys: tuple[str, ...] = ("gib", "dppl")
                 ) -> list[dict]:
    """返回非支配点子集，按 ``keys`` 字典序升序，并去除重复点。

    重复判定取 ``keys`` 各维度的 6 位小数：浮点噪声会让同一点产生细微差异的副本，
    直接比较原始元组会把它们都保留下来。

    Args:
        points: 候选点列表，每项须包含 ``keys`` 中的全部维度。
        keys: 参与比较的维度名，默认 ``("gib", "dppl")``，均为越小越好。

    Returns:
        非支配点列表，按 ``keys`` 升序排列，已按 6 位小数去重。

    Raises:
        KeyError: 某个点缺少 ``keys`` 中的某个维度。

    Note:
        复杂度 O(n²)，n 为候选点数；预算扫描通常只产出几十个点，无需优化。
    """
    front = []
    for i, p in enumerate(points):
        if any(dominates(q, p, keys) for j, q in enumerate(points) if i != j):
            continue
        front.append(p)
    seen, uniq = set(), []
    for p in sorted(front, key=lambda x: tuple(x[k] for k in keys)):
        key = tuple(round(p[k], 6) for k in keys)
        if key not in seen:
            seen.add(key)
            uniq.append(p)
    return uniq


def to_points(sweep: list[dict], extra: list[dict] | None = None) -> list[dict]:
    """把预算扫描结果与额外对照点合并为统一的绘图点列表。

    Args:
        sweep: 预算扫描结果，每项须含 ``gib``（单位 GiB）、``dppl``（单位百分比）
            与 ``budget``（单位 GiB，仅用于生成标签）。
        extra: 额外对照点，需自带 ``gib`` / ``dppl`` / ``label`` 字段，例如带
            ``mark`` 的关键点；为 ``None`` 时只返回扫描点。

    Returns:
        ``{gib, dppl, label}`` 记录列表，``gib`` 单位 GiB，``dppl`` 单位百分比。

    Raises:
        KeyError: ``sweep`` 中某项缺少 ``gib`` / ``dppl`` / ``budget``。
    """
    pts = [{"gib": r["gib"], "dppl": r["dppl"],
            "label": f"budget {r['budget']:.2f} GiB"} for r in sweep]
    if extra:
        pts += extra
    return pts


def plot(points: list[dict], front: list[dict], out_png: str,
         title: str = "Pareto front: memory vs accuracy loss") -> None:
    """绘制 Pareto 图并写出 PNG 文件。

    图中包含候选点散点、前沿折线，以及带 ``mark`` 字段的关键点标注。

    Args:
        points: 全部候选点，每项须含 ``gib``（单位 GiB）与 ``dppl``（单位百分比）。
        front: 前沿点，通常为 :func:`pareto_front` 的返回值。
        out_png: 输出图片路径。
        title: 图标题。

    Returns:
        ``None``。函数以 150 dpi 写出图片，并打印保存路径。

    Raises:
        KeyError: 某个点缺少 ``gib`` 或 ``dppl``。
        OSError: ``out_png`` 不可写。

    Note:
        默认标签使用英文：matplotlib 缺失中文字形时，中文标签会渲染为方块。
        ``Agg`` 后端不依赖图形环境，可在无显示器的机器上运行。
    """
    # 延迟导入：matplotlib 体积大且非出图路径所必需，保持模块可在轻量环境导入。
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 5))
    # 以 6 位小数作为点身份键，与 pareto_front 的去重口径保持一致。
    fs = {(round(p["gib"], 6), round(p["dppl"], 6)) for p in front}

    others = [p for p in points
              if (round(p["gib"], 6), round(p["dppl"], 6)) not in fs
              and not p.get("mark")]
    if others:
        ax.scatter([p["gib"] for p in others], [p["dppl"] for p in others],
                   s=24, c="#9aa5b1", label="candidate plans", zorder=2)
    if front:
        ax.plot([p["gib"] for p in front], [p["dppl"] for p in front],
                "-o", c="#1f77b4", ms=6, label="Pareto front", zorder=3)

    for p in points:
        if p.get("mark"):
            ax.scatter([p["gib"]], [p["dppl"]], marker=p["mark"], s=150,
                       zorder=4, label=p.get("label", ""))
            ax.annotate(p.get("label", ""), (p["gib"], p["dppl"]),
                        textcoords="offset points", xytext=(7, 5), fontsize=8)

    ax.set_xlabel("weight memory (GiB)")
    ax.set_ylabel("total dPPL (%)")
    ax.set_title(title)
    ax.grid(True, alpha=.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    print(f"saved {out_png}")

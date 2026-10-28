"""AlphaQ 的免校准重要性指标 PL_Alpha_Hill。

基于 HT-SR 理论：训练越充分，权重相关矩阵特征值谱的尾部衰减越慢，
对应的 Hill 指数 α 越小。计算流程为：

- 取权重矩阵 ``W``，构造相关矩阵 ``X = WᵀW``
- 以 ``X`` 的特征值谱作为经验谱密度（ESD）
- 用 Hill 估计器拟合谱尾部：``α = 1 + k / Σ ln(λ_i / λ_k)``，λ 降序
- FARMS 采样：按固定长宽比截取子块，消除矩阵形状对谱的影响

公开 API：

- :func:`block_eigs`：子块采样与 SVD，返回每块的 ``λ = σ²``
- :func:`alpha_hill`：单条谱上的朴素 Hill 估计
- :func:`alpha_pooled` / :func:`alpha_median`：拼接池与逐块中位两种汇总路径
- :func:`spectral_stats`：遍历模型全部 ``nn.Linear``，汇总 (α, 方差)
- :func:`save_spectral_csv` / :func:`load_spectral_csv`：CSV 序列化
- :func:`eta_cost`：AlphaQ 的层代价模型

Note:
    在 Qwen2.5-1.5B 的 196 个模块上实测，α 与量化损伤的相关性极弱：
    与单位参数损伤 ρ ≈ +0.01（12 种实现变体均接近 0）；与总损伤 ρ ≈ +0.42，
    但参数量单独就有 +0.43，且 ρ(α, 参数量) = +0.52；同形状内部（每类 28 个）
    ``|ρ| ≤ 0.22`` 且符号杂乱。本模块因此仅作候选策略之一，不作为唯一依据。
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch

__all__ = ["block_eigs", "alpha_hill", "alpha_pooled", "alpha_median",
           "spectral_stats", "save_spectral_csv", "load_spectral_csv"]

_DFLT = dict(sub=128, stride=128, max_blocks=256, k_frac=0.1)


@torch.no_grad()
def block_eigs(W: torch.Tensor, sub: int = 128, stride: int = 128,
               max_blocks: int = 256) -> list[torch.Tensor]:
    """按 FARMS 规则截取子块并做 SVD，返回每块的 ``λ = σ²``。

    Args:
        W: 权重张量；多于二维时按首维展平，稀疏张量先转稠密。
        sub: 子块边长，单位元素个数，子块为 ``sub × sub``。
        stride: 子块采样步长，单位元素个数。
        max_blocks: 子块数量上限；超出时按固定随机种子 0 无放回采样。

    Returns:
        λ 张量列表，每项为一子块的奇异值平方，按降序排列；
        任一边长小于 4 的子块被丢弃，因此返回列表可能短于采样块数。

    Note:
        用 SVD 奇异值的平方而不是显式构造 ``WᵀW``：条件数不被平方，
        特征值动态范围更小，数值更稳定。
        权重先在 CPU 上以 float32 复制后计算，不占用显存，也不修改原张量。

    Warning:
        固定随机种子 0 使同一形状的结果可复现；改动种子会改变
        ``max_blocks`` 截断后的子块集合，进而改变 α。
    """
    W = W.detach().float().cpu()
    if W.is_sparse:
        W = W.to_dense()
    if W.dim() > 2:
        W = W.reshape(W.shape[0], -1)
    m, n = W.shape
    idx = [(i, j) for i in range(0, max(m - sub, 0) + 1, stride)
           for j in range(0, max(n - sub, 0) + 1, stride)]
    if not idx:
        idx = [(0, 0)]
    if len(idx) > max_blocks:
        g = torch.Generator().manual_seed(0)
        keep = torch.randperm(len(idx), generator=g)[:max_blocks].tolist()
        idx = [idx[k] for k in sorted(keep)]
    out = []
    for (i, j) in idx:
        blk = W[i:i + sub, j:j + sub]
        if min(blk.shape) < 4:
            continue
        out.append(torch.linalg.svdvals(blk) ** 2)   # NOTE: 已是降序，无需翻转
    return out


def alpha_hill(lam_desc: torch.Tensor, k_frac: float = 0.1) -> float:
    """取降序谱中最大的 ``k_frac`` 比例尾部，用朴素 Hill 估计器拟合 α。

    Args:
        lam_desc: 降序排列的特征值或奇异值平方序列。
        k_frac: 参与拟合的尾部比例，无量纲；``0.1`` 表示取前 10%。

    Returns:
        Hill 估计的 α，无量纲；序列长度不足 4 或分母过小时返回 ``nan``。

    Note:
        k 的下界为 ``max(10, int(n * k_frac))``、上界为 ``n - 1``，
        上界保证分母至少含一个非零的 ``ln(λ_i / λ_k)`` 项。
    """
    n = lam_desc.numel()
    if n < 4:
        return float("nan")
    k = max(1, min(max(10, int(n * k_frac)), n - 1))
    ref = float(lam_desc[k - 1].clamp_min(1e-12))
    top = lam_desc[:k].clamp_min(1e-12)
    denom = float(torch.log(top / ref).sum())
    return float("nan") if denom <= 1e-12 else 1.0 + k / denom


def alpha_pooled(blocks, k_frac: float = 0.1) -> float:
    """把所有子块的 λ 拼接成池后排序，在池上拟合 α。

    Args:
        blocks: 各子块的 λ 张量序列，单个张量内部应为降序。
        k_frac: 参与拟合的尾部比例，无量纲。

    Returns:
        拼接池上的 Hill α；``blocks`` 为空时返回 ``nan``。
    """
    if not blocks:
        return float("nan")
    pool, _ = torch.sort(torch.cat(blocks), descending=True)
    return alpha_hill(pool, k_frac)


def alpha_median(blocks, k_frac: float = 0.1) -> float:
    """逐块估计 α 后取中位数，保留子块间的差异。

    Args:
        blocks: 各子块的 λ 张量序列，单个张量内部应为降序。
        k_frac: 参与拟合的尾部比例，无量纲。

    Returns:
        各块 α 的中位数；全部块估计失败时返回 ``nan``。

    Note:
        与拼接池做法相比更保守：单个重尾子块不会主导整层估计。
    """
    v = [alpha_hill(b, k_frac) for b in blocks]
    v = [x for x in v if not np.isnan(x)]
    return float(np.median(v)) if v else float("nan")


def spectral_stats(model, use_pooled: bool = True, **kw) -> dict:
    """遍历模型中所有 ``nn.Linear``，计算每层的 α 与权重方差。

    Args:
        model: 已加载的 PyTorch 模型，按 ``named_modules`` 遍历。
        use_pooled: 为 ``True`` 时用拼接池估计 α，否则用逐块中位数。
        **kw: 覆盖默认参数，可含 ``sub`` / ``stride`` / ``max_blocks`` / ``k_frac``。

    Returns:
        ``{name: {"alpha": float, "var": float, "params": int}}``；
        ``params`` 为权重元素个数，非打包后的字节数。

    Note:
        复杂度约 O(Σ numel(W) · k)，k 为单层子块数，耗时由 SVD 主导。
        权重以 float32 在 CPU 上读取，不修改模型状态。
    """
    kw = {**_DFLT, **kw}
    out = {}
    for name, mod in model.named_modules():
        if not isinstance(mod, torch.nn.Linear):
            continue
        W = mod.weight.data
        blocks = block_eigs(W, kw["sub"], kw["stride"], kw["max_blocks"])
        a = (alpha_pooled(blocks, kw["k_frac"]) if use_pooled
             else alpha_median(blocks, kw["k_frac"]))
        out[name] = {
            "alpha": a,
            "var": float(W.detach().float().var()),
            "params": int(W.numel()),
        }
    return out


def save_spectral_csv(stats: dict, path: str) -> None:
    """把 :func:`spectral_stats` 的结果写成 CSV。

    Args:
        stats: :func:`spectral_stats` 返回的字典。
        path: 输出文件路径，UTF-8 编码；父目录须已存在。

    Raises:
        OSError: 路径不可写或父目录不存在。

    Warning:
        目标文件被整体覆盖，原内容不可恢复。
    """
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["layer_name", "alpha", "variance", "params"])
        for n, s in stats.items():
            w.writerow([n, round(s["alpha"], 6), round(s["var"], 8), s["params"]])


def load_spectral_csv(path: str) -> dict:
    """读取 :func:`save_spectral_csv` 写出的 CSV。

    Args:
        path: CSV 文件路径，UTF-8 编码，须含 ``layer_name`` / ``alpha`` /
            ``variance`` / ``params`` 四列。

    Returns:
        ``{layer_name: {"alpha": float, "var": float, "params": int}}``。

    Raises:
        KeyError: CSV 缺少上述任一列。
        FileNotFoundError: 路径不存在。
    """
    out = {}
    for r in csv.DictReader(open(path, encoding="utf-8")):
        out[r["layer_name"]] = {
            "alpha": float(r["alpha"]),
            "var": float(r["variance"]),
            "params": int(r["params"]),
        }
    return out


# ============================================================ AlphaQ 目标函数
def eta_cost(params: int, var: float, alpha: float, alpha_bar: float,
             gamma: float, bits: int) -> float:
    """计算 AlphaQ 的层代价 ``η = (ᾱ/α)^γ · Var(W) · 2^{-2b}``。

    Args:
        params: 该层参数量，单位个；本式不使用，仅为统一调用签名保留。
        var: 该层权重方差。
        alpha: 该层的 Hill 指数 α，无量纲，下限按 ``1e-9`` 截断。
        alpha_bar: 全模型 α 的参考值，无量纲。
        gamma: 幂次放大系数，无量纲。
        bits: 目标位宽，单位 bits；每减少 1 bit，代价乘 4。

    Returns:
        该层相对代价，无量纲，数值越小表示预期损伤越低。

    Warning:
        该式在 1.5B 模型上与实测方向相反：η 越高，实测损伤反而越低
        （ρ = −0.42）。仅可用于层间排序参考，不可作为最终结论。
    """
    return (alpha_bar / max(alpha, 1e-9)) ** gamma * var * (2.0 ** (-2 * bits))

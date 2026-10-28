"""
aq.cost — 显存、卡型与年成本的估算。

三个量依次推导：

1) 显存需求
    M = W(权重) + KV(KV cache) + overhead

    W    = Σ params_i × bytes(w_i)          ← 随量化下降
    KV   = 2 · L · n_kv · d_head · T · b · bytes_kv
    overhead = CUDA context + 缓存池 + 激活（经验值 ~0.5 GiB）

2) 最小卡型
    min_gpu(M) = 显存 >= M 的最便宜一张卡

3) 年成本
    年成本 = 卡数 × 单价(元/时) × 年运行小时数
    节省   = 年成本(bf16 方案) − 年成本(量化方案)

已知局限：

- KV cache 不随权重量化缩小，因此量化的显存收益存在上限。上下文越长，
  收益占比越低：1.5B、T=32768、b=1 时权重由 2.88 GiB 降到 0.72 GiB，
  而 KV cache 恒为 0.88 GiB，总节省 2.16 / 3.76 = 58%，而非 75%。
- 「装得下」不等于「跑得快」。量化让模型可以落到更便宜的卡上，这是显存
  维度的实测结论；吞吐提升还需要带宽充足且软件开销低的环境（4090 + vLLM）。
  本模块只覆盖显存与成本，不涉及速度承诺。
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ============================================================================
# 卡型库
# ============================================================================
# 价格为占位值 None：单价随云商、地区与计费方式（按量 / 包月）差异很大，
# 须填入实际询价结果后才能得出成本结论。
# ============================================================================


@dataclass(frozen=True)
class GpuTier:
    name: str
    mem_gib: float
    price_per_hour: float | None = None   # 元/小时；None = 待填


GPU_TIERS: list[GpuTier] = [
    GpuTier("消费卡 4GB",   4,   None),
    GpuTier("消费卡 8GB",   8,   None),
    GpuTier("消费卡 12GB",  12,  None),
    GpuTier("T4 16GB",      16,  None),
    GpuTier("RTX 4090 24GB", 24, None),
    GpuTier("A10 24GB",     24,  None),
    GpuTier("L4 24GB",      24,  None),
    GpuTier("A100 40GB",    40,  None),
    GpuTier("A100 80GB",    80,  None),
]


# ============================================================================
# 1) 显存三部分
# ============================================================================
def weight_gib(n_params: int, bits: float) -> float:
    """按位宽估算权重显存，单位 GiB。

    Args:
        n_params: 参数量。
        bits: 位宽，可以是 16 / 8 / 4，也可以是混合精度方案算出的平均位宽。

    Returns:
        权重显存，单位 GiB。

    Raises:
        ValueError: ``n_params`` 或 ``bits`` 不大于 0。

    Note:
        理论值。blockwise 量化需额外存 scale / zero-point，实测会略高，
        bnb 4bit 约为理论值的 1.05–1.10 倍。
    """
    if n_params <= 0:
        raise ValueError(f"n_params 必须 > 0，收到 {n_params}")
    if bits <= 0:
        raise ValueError(f"bits 必须 > 0，收到 {bits}")
    return n_params * bits / 8 / 1024 ** 3


def kv_cache_gib(num_layers: int, n_kv_heads: int, head_dim: int,
                 seq_len: int, batch: int = 1, bytes_per_elem: int = 2) -> float:
    """估算 KV cache 显存，单位 GiB。

        KV = 2(K 和 V) · L · n_kv · d_head · T · b · bytes

    Args:
        num_layers: transformer 层数 ``L``。
        n_kv_heads: KV 头数 ``n_kv``；GQA 下远小于 Q 头数，不能传 Q 头数。
        head_dim: 每个头的维度 ``d_head``。
        seq_len: 上下文长度 ``T``，单位为 token。
        batch: 并发批大小 ``b``。
        bytes_per_elem: 每个 KV 元素的字节数，bf16 为 2。

    Returns:
        KV cache 显存，单位 GiB。

    Note:
        该项与权重量化无关，不随权重位宽下降。
        参考量级：Qwen2.5-1.5B（L=28, n_kv=2, d_head=128, bf16）约 28 KB/token。
    """
    total_bytes = 2 * num_layers * n_kv_heads * head_dim * seq_len * batch * bytes_per_elem
    return total_bytes / 1024 ** 3


def total_gib(n_params: int, bits: float, num_layers: int, n_kv_heads: int,
              head_dim: int, seq_len: int, batch: int = 1,
              overhead_gib: float = 0.5) -> dict:
    """汇总总显存，单位 GiB。

    返回分项明细，便于逐项核对而非只看总数。

    Args:
        n_params: 参数量。
        bits: 权重位宽，含义同 :func:`weight_gib`。
        num_layers: transformer 层数。
        n_kv_heads: KV 头数。
        head_dim: 每个头的维度。
        seq_len: 上下文长度，单位为 token。
        batch: 并发批大小。
        overhead_gib: CUDA context、缓存池与激活的显存开销，单位 GiB。

    Returns:
        字典，包含 ``weight_gib`` / ``kv_cache_gib`` / ``overhead_gib`` /
        ``total_gib`` 四项。
    """
    w = weight_gib(n_params, bits)
    kv = kv_cache_gib(num_layers, n_kv_heads, head_dim, seq_len, batch)
    return {
        "weight_gib": w,
        "kv_cache_gib": kv,
        "overhead_gib": overhead_gib,
        "total_gib": w + kv + overhead_gib,
    }


# ============================================================================
# 2) 卡型选择
# ============================================================================
DEFAULT_HEADROOM = 0.8
"""
余量系数：模型最多占卡显存的 80%。

把显存用到 100% 在工程上不可行，原因有三：

  · 需要给 KV cache 增长、激活与显存碎片留空间；
  · vLLM 自身默认只用到 90%（``gpu_memory_utilization=0.9``）；
  · 显存接近满时，Windows WDDM 会溢出到共享内存（见 env.md），速度下降约 7 倍。

因此「能装下」的判据是 ``total / 0.8 <= 卡显存``。
"""


def min_gpu(total_gib_needed: float, tiers: list[GpuTier] | None = None,
            headroom: float = DEFAULT_HEADROOM) -> GpuTier:
    """返回能装下的最小卡。

    Args:
        total_gib_needed: 模型总显存需求，单位 GiB。
        tiers: 候选卡型列表，默认取 ``GPU_TIERS``。
        headroom: 显存占用上限比例；默认 0.8 表示最多用卡显存的 80%。

    Returns:
        满足需求的最小卡型；同显存档位取列表中靠前者。

    Raises:
        ValueError: ``headroom`` 不在 ``(0, 1]`` 内，或没有单卡能装下。

    Note:
        ``headroom=1.0`` 表示允许用满显存，属理论值，不适用于对外报告。
    """
    if not 0 < headroom <= 1.0:
        raise ValueError(f"headroom 必须在 (0, 1] 内，收到 {headroom}")
    need = total_gib_needed / headroom
    tiers = sorted(tiers or GPU_TIERS, key=lambda g: g.mem_gib)
    for g in tiers:
        if g.mem_gib >= need:
            return g
    raise ValueError(
        f"没有单卡能装下 {total_gib_needed:.2f} GiB"
        f"（按 {headroom:.0%} 余量需 {need:.2f} GiB）；需要多卡或更大卡")


def num_gpus_if_single_limit(total_gib_needed: float, gpu: GpuTier) -> int:
    """估算限定单卡型号时所需张数。

    按显存线性向上取整，未计入并行通信开销与冗余。

    Args:
        total_gib_needed: 模型总显存需求，单位 GiB。
        gpu: 限定的卡型。

    Returns:
        所需卡数，至少为 1。
    """
    import math
    return max(1, math.ceil(total_gib_needed / gpu.mem_gib))


# ============================================================================
# 3) 年成本与节省
# ============================================================================
def annual_cost(gpu: GpuTier, hours_per_year: float = 720 * 12,
                n_gpus: int = 1) -> float:
    """估算年成本，单位元。

    Args:
        gpu: 卡型，其 ``price_per_hour`` 必须已填。
        hours_per_year: 年运行小时数，默认按每月 720 小时 × 12 计。
        n_gpus: 卡数。

    Returns:
        年成本，单位元。

    Raises:
        ValueError: ``gpu.price_per_hour`` 为 ``None``，即单价未填。
    """
    if gpu.price_per_hour is None:
        raise ValueError(
            f"卡型 {gpu.name} 的单价未填。\n"
            f"请在 GPU_TIERS 里填入你【实际询到】的价格（元/小时），\n"
            f"并注明来源与日期 —— 答辩会被问'这价格哪来的'。"
        )
    return n_gpus * gpu.price_per_hour * hours_per_year


def compare(bf16_total: float, quant_total: float, gpu: GpuTier | None = None,
            hours_per_year: float = 720 * 12,
            headroom: float = DEFAULT_HEADROOM) -> dict:
    """对比 bf16 与量化方案的成本。

    两种口径都给出：

    ① 同模型下省多少钱（卡型降档）；
    ② 同成本下能跑多大的模型，由调用方按模型表用 :func:`fit_matrix` 推算。

    Args:
        bf16_total: bf16 方案的总显存，单位 GiB。
        quant_total: 量化方案的总显存，单位 GiB。
        gpu: 保留参数，当前实现未使用。
        hours_per_year: 年运行小时数，默认按每月 720 小时 × 12 计。
        headroom: 显存占用上限比例。

    Returns:
        字典，包含 ``headroom`` / ``bf16`` / ``quant`` / ``mem_saving_gib`` /
        ``mem_saving_pct``（百分比）/ ``gpu_downgraded`` / ``annual``。
        单价未填时 ``annual`` 为 ``None``，不给出成本结论。
    """
    g_bf16 = min_gpu(bf16_total, headroom=headroom)
    g_quant = min_gpu(quant_total, headroom=headroom)
    out = {
        "headroom": headroom,
        "bf16": {
            "total_gib": bf16_total,
            "gpu": g_bf16.name,
            "mem_gib": g_bf16.mem_gib,
        },
        "quant": {
            "total_gib": quant_total,
            "gpu": g_quant.name,
            "mem_gib": g_quant.mem_gib,
        },
        "mem_saving_gib": bf16_total - quant_total,
        "mem_saving_pct": (bf16_total - quant_total) / bf16_total * 100 if bf16_total else 0.0,
        "gpu_downgraded": g_bf16.mem_gib != g_quant.mem_gib,
    }
    if g_bf16.price_per_hour is not None and g_quant.price_per_hour is not None:
        c_bf16 = annual_cost(g_bf16, hours_per_year)
        c_quant = annual_cost(g_quant, hours_per_year)
        out["annual"] = {
            "bf16": c_bf16,
            "quant": c_quant,
            "saving": c_bf16 - c_quant,
            "saving_pct": (c_bf16 - c_quant) / c_bf16 * 100,
            "hours_per_year": hours_per_year,
        }
    else:
        out["annual"] = None   # 单价未填，不出成本结论
    return out


# ============================================================================
# 4) 模型规模 × 精度 → 卡型矩阵
# ============================================================================
# 参数量按常见规模估算；实际用 model.num_parameters() 替换更准。
MODEL_SIZES = {
    "0.5B": 494_000_000,
    "1.5B": 1_543_714_304,      # 本实现实测值
    "3B":   3_090_000_000,
    "7B":   7_620_000_000,
    "13B":  13_020_000_000,
    "32B":  32_760_000_000,
}

BITS = {"FP16": 16, "INT8": 8, "INT4": 4}


def fit_matrix(models: dict | None = None,
               bits_map: dict | None = None,
               seq_len: int = 2048,
               headroom: float = DEFAULT_HEADROOM) -> list[dict]:
    """生成「模型规模 × 精度 → 最小卡型」矩阵。

    Args:
        models: 模型名到参数量的映射，默认取 ``MODEL_SIZES``。
        bits_map: 精度名到位宽的映射，默认取 ``BITS``。
        seq_len: 估算 KV cache 时采用的上下文长度，单位为 token；
            传 0 则只计权重，结果偏激进。
        headroom: 显存占用上限比例，默认 0.8，即模型最多占卡显存的 80%。

    Returns:
        行列表，每行含 ``model`` / ``n_params``，以及每种精度的
        ``{精度}_gib``（仅权重）、``{精度}_total``、``{精度}_gpu``。

    Note:
        默认计入 ``seq_len=2048`` 的 KV cache，比只算权重更接近真实部署。
        KV cache 参数固定取 1.5B 的配置（L=28, n_kv=2, d_head=128）作占位，
        换模型时须替换为该模型的真实值。
        ``headroom=1.0`` 用于观察理论极限，对外报告应使用默认值。
    """
    models = models or MODEL_SIZES
    bits_map = bits_map or BITS
    rows = []
    for mname, n_params in models.items():
        row = {"model": mname, "n_params": n_params}
        for bname, bits in bits_map.items():
            w = weight_gib(n_params, bits)
            kv = kv_cache_gib(28, 2, 128, seq_len) if seq_len else 0.0
            total = w + kv + 0.5
            g = min_gpu(total, headroom=headroom)
            row[f"{bname}_gib"] = round(w, 2)
            row[f"{bname}_total"] = round(total, 2)
            row[f"{bname}_gpu"] = g.name
        rows.append(row)
    return rows


def print_fit_matrix(rows: list[dict] | None = None,
                     headroom: float = DEFAULT_HEADROOM) -> str:
    """把矩阵渲染成 markdown 表。

    Args:
        rows: :func:`fit_matrix` 的返回值，默认内部现算。
        headroom: 传给 :func:`fit_matrix` 的显存占用上限比例。

    Returns:
        可直接贴入报告的 markdown 文本。
    """
    rows = rows or fit_matrix(headroom=headroom)
    head = ("| 模型 | FP16 权重 | FP16 最低卡 | INT8 权重 | INT8 最低卡 "
            "| INT4 权重 | INT4 最低卡 |")
    sep = "|---|---|---|---|---|---|---|"
    lines = [head, sep]
    for r in rows:
        lines.append(
            f"| {r['model']} | {r['FP16_gib']} GiB | {r['FP16_gpu']} "
            f"| {r['INT8_gib']} GiB | {r['INT8_gpu']} "
            f"| {r['INT4_gib']} GiB | {r['INT4_gpu']} |"
        )
    lines.append("")
    lines.append(f"> 判据：模型总占用（权重 + KV cache@T=2048 + 0.5 GiB 余量）"
                 f"不超过卡显存的 {headroom:.0%}。")
    return "\n".join(lines)

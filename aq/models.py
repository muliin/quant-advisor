"""aq/models.py — 常见模型的知识表与名称模糊匹配。

维护常见开源模型的参数规模知识表，用于从自由文本中识别模型并按位宽
估算显存占用。公开 API：

- :data:`MODELS`：模型名到别名、参数量、层数、专家配置的映射
- :func:`match_model`：从自由文本中解析模型，未命中返回 ``None``
- :func:`describe_model`：按位宽估算显存需求与可装载性
- :func:`installed_vram_need`：按位宽估算加载所需显存

MoE 模型需区分两个口径：``total_b`` 为总参数量，决定显存，所有权重
必须驻留；``active_b`` 为每 token 激活参数量，决定算力。DeepSeek-V3
为 671B 总 / 37B 激活，其 4bit 需求超出 16 GiB 显存约 20 倍。

已知局限：知识表为静态数据，未收录的模型按未命中处理，由调用方回退
到交互式询问。
"""
from __future__ import annotations

import re

__all__ = ["MODELS", "match_model", "describe_model", "installed_vram_need"]

# name -> (别名列表, 总参B, 激活参B(0=dense), 层数, 专家数, top_k)
MODELS = {
    "deepseek-v3": (["deepseekv3", "deepseek-v3", "deepseek v3", "v3", "dsv3",
                      "deepseek3"], 671.0, 37.0, 61, 256, 8),
    "deepseek-r1": (["deepseekr1", "deepseek-r1", "r1", "deepseek r1"],
                    671.0, 37.0, 61, 256, 8),
    "deepseek-v2-lite": (["deepseekv2lite", "deepseek-v2-lite", "v2-lite"],
                         15.7, 2.4, 27, 64, 6),
    "qwen3-235b-a22b": (["qwen3-235b", "qwen3 235b", "235b-a22b"],
                        235.0, 22.0, 94, 128, 8),
    "qwen3-30b-a3b": (["qwen3-30b", "qwen3 30b", "30b-a3b", "qwen3moe"],
                      30.5, 3.3, 48, 128, 8),
    "qwen1.5-moe-a2.7b": (["qwen1.5-moe", "qwen1.5moe", "a2.7b"],
                         14.3, 2.7, 24, 60, 4),
    "olmoe": (["olmoe", "olmo-e", "olmoe-1b-7b"], 6.92, 1.18, 16, 64, 8),
    "mixtral-8x7b": (["mixtral", "mixtral-8x7b", "8x7b"], 46.7, 12.9, 32, 8, 2),
    "qwen2.5-72b": (["qwen2.5-72b", "qwen-72b", "72b"], 72.7, 0.0, 80, 0, 0),
    "qwen2.5-32b": (["qwen2.5-32b", "32b"], 32.8, 0.0, 64, 0, 0),
    "qwen2.5-14b": (["qwen2.5-14b", "14b"], 14.8, 0.0, 48, 0, 0),
    "qwen2.5-7b": (["qwen2.5-7b", "qwen-7b"], 7.6, 0.0, 28, 0, 0),
    "qwen2.5-3b": (["qwen2.5-3b", "qwen-3b"], 3.1, 0.0, 36, 0, 0),
    "qwen2.5-1.5b": (["qwen2.5-1.5b", "qwen-1.5b"], 1.54, 0.0, 28, 0, 0),
    "llama-3.1-70b": (["llama-3.1-70b", "llama3.1-70b", "llama-70b"],
                      70.6, 0.0, 80, 0, 0),
    "llama-3.1-8b": (["llama-3.1-8b", "llama3.1-8b", "llama-8b", "llama3-8b"],
                     8.03, 0.0, 32, 0, 0),
    "llama-3.2-3b": (["llama-3.2-3b", "llama3.2-3b"], 3.21, 0.0, 28, 0, 0),
    "mistral-7b": (["mistral-7b", "mistral"], 7.24, 0.0, 32, 0, 0),
    "gemma-2-9b": (["gemma-2-9b", "gemma2-9b", "gemma"], 9.24, 0.0, 42, 0, 0),
    "phi-3-mini": (["phi-3-mini", "phi3", "phi-3"], 3.82, 0.0, 32, 0, 0),
}


def match_model(text: str) -> dict | None:
    """从自由文本中解析出匹配别名最长的已知模型。

    遍历 :data:`MODELS` 中每个模型名及其全部别名，在文本中做子串匹配，
    取命中别名最长者。

    Args:
        text: 待解析文本。大小写不敏感，下划线先归一化为连字符。

    Returns:
        命中时返回字典，含 ``name``、``matched``（命中的别名）、
        ``total_b``、``active_b``、``layers``、``experts``、``top_k``、
        ``is_moe``；未命中返回 ``None``。

    Note:
        采用最长匹配而非首个匹配，否则 ``r1`` 会命中英文单词，
        ``qwen2.5-7b`` 也会被 ``7b`` 抢先。
    """
    t = text.lower().replace("_", "-")
    best = None
    for name, (aliases, total, active, layers, experts, topk) in MODELS.items():
        cands = [name.lower()] + [a.lower() for a in aliases]
        for a in cands:
            if a in t:
                if best is None or len(a) > len(best["matched"]):
                    best = {"name": name, "matched": a, "total_b": total,
                            "active_b": active, "layers": layers,
                            "experts": experts, "top_k": topk,
                            "is_moe": experts > 0}
    # 至少 3 个字符才算数（防止 "v3"、"r1" 误命中）
    if best and len(best["matched"]) < 3:
        # 只有形如 "v3"/"r1" 且上下文有 deepseek 才认
        if not re.search(r"deepseek", t):
            return None
    return best


def describe_model(m: dict, vram_gib: float | None = None) -> dict:
    """按位宽估算模型的显存需求，并判断能否装入给定显存。

    Args:
        m: :func:`match_model` 返回的模型字典，须含 ``total_b`` 与
            ``active_b``。
        vram_gib: 可用显存，单位 GiB。为 ``None`` 时只估算需求，不做
            可装载性判断。

    Returns:
        字典，含 ``name``、``total_b``、``active_b``、``is_moe``、
        ``ratio``（总参 / 激活参，保留 2 位小数，dense 模型为 ``None``）
        与 ``by_bits``。``by_bits`` 以位宽为键，每项含 ``bits``、``gib``
        （需求，单位 GiB）；给出 ``vram_gib`` 时另含 ``fits`` 与
        ``short_by``（需求与可用显存之比，单位倍，装得下时为 ``None``）。

    Note:
        显存按 ``总参数量 × 每参比特数 / 8`` 估算，另加 0.434 GiB 固定
        开销。4bit 按 4.128 bits/参数计（含量化 scale 开销），其余位宽
        按名义值计。``fits`` 要求需求不超过可用显存的 85%。
    """
    total = m["total_b"] * 1.029e9
    out = {"name": m["name"], "total_b": m["total_b"],
           "active_b": m["active_b"], "is_moe": m["is_moe"],
           "ratio": round(m["total_b"] / m["active_b"], 2) if m["active_b"] else None,
           "by_bits": {}}
    for b, eff in ((16, 16.0), (8, 8.0), (4, 4.128), (3, 3.0)):
        gib = total * eff / 8 / (1024 ** 3) + 0.434
        row = {"bits": b, "gib": round(gib, 2)}
        if vram_gib:
            row["fits"] = gib <= vram_gib * 0.85
            row["short_by"] = (round(gib / vram_gib, 1)
                               if gib > vram_gib else None)
        out["by_bits"][b] = row
    return out


def installed_vram_need(m: dict, bits: int = 4) -> float:
    """估算模型在给定位宽下的权重显存需求。

    Args:
        m: :func:`match_model` 返回的模型信息字典。
        bits: 目标位宽，支持 16 / 8 / 4 / 3；4bit 按 4.128 bits/参数计
            （含分组 scale 的额外开销）。

    Returns:
        权重显存，单位 GiB，已包含 0.434 GiB 的固定开销
        （``lm_head`` 等不量化部分）。

    Note:
        仅统计权重，不含 KV cache 与激活，因此是下界。
    """
    eff = {16: 16.0, 8: 8.0, 4: 4.128, 3: 3.0}.get(bits, float(bits))
    return m["total_b"] * 1.029e9 * eff / 8 / (1024 ** 3) + 0.434

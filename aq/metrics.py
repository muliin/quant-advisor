"""PPL 评测指标。

本模块定义评测接口：载入评测文本、对 token 序列计算 PPL、按
bf16 / int8 / int4 三配置对比并渲染成 markdown 表格。
参考实现位于 ``evaluate.py`` 与 ``evaluate_参考实现.py``，此处只保留
其他模块依赖的签名。

PPL 定义为 ``exp(交叉熵)``。位置 i 的 logits 预测位置 i+1 的 token，
因此计算时须对 logits 做右移：``logits[:-1]`` 对 ``labels[1:]``。

Note:
    序列必须按固定 ``chunk`` 切块；chunk 长度会显著影响 PPL
    （实测 128 与 512 相差 2.3 倍），不同 chunk 长度下的结果不可比较。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

import math
from pathlib import Path


CHUNK = 512
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# ============================================================================
# 接口（实现时保持签名不变，其他模块依赖它）
# ============================================================================




def _dtype_kwarg(dtype: torch.dtype) -> dict:
    import transformers
    major = int(transformers.__version__.split(".")[0])
    return {"dtype": dtype} if major >=5 else {"torch_dtype": dtype}

def load_text(lang: str = "en") -> str:
    """载入指定语言的评测文本。

    ``en`` 读取 ``data/en_eval.txt``（WikiText-2），``zh`` 读取
    ``data/zh_eval.txt``，后者须为不重复的真实长文本。

    Args:
        lang: 语言标识，``"en"`` 或 ``"zh"``。

    Returns:
        评测文本全文。

    Raises:
        FileNotFoundError: 对应的评测文本文件不存在。
        ValueError: 文本短于 1500 字符，PPL 不可信；或前后半段完全相同，
            判定为重复拼接，PPL 会失真。
    """
    p = DATA_DIR / ("en_eval.txt" if lang == "en" else "zh_eval.txt")
    if not p.exists():
        raise FileNotFoundError(f"缺少评测文本: {p}")
    t = p.read_text(encoding="utf-8")    
    if len(t) < 1500:
        raise ValueError(f"{p} 太短（{len(t)} 字符），PPL 不可信")
    half = len(t) // 2
    if t[:half] == t[half:2 * half]:
        raise ValueError(f"{p} 疑似重复拼接，PPL 会失真")
    return t
    
def ppl_from_ids(model, ids: torch.Tensor, chunk: int = 512,
                 shift: bool = True, max_blocks: int | None = None) -> tuple[float, int]:
    """对一段 token 序列计算困惑度（PPL）。

    按固定 ``chunk`` 切块，逐块累加交叉熵后做宏平均。``shift`` 为真时
    用 ``logits[:-1]`` 预测 ``labels[1:]``；块间不共享上下文，
    因此块长会显著影响结果。

    Args:
        model: 已调用 ``.eval()`` 的因果语言模型。
        ids: 一维 LongTensor 形式的 token id，长度须大于 ``2 * chunk``。
        chunk: 每块的 token 数；所有对比实验必须取同一值。
        shift: 是否做位置右移。``True`` 为正确实现；``False`` 仅用于自检，
            此时 logits 与 label 的配对是失衡的。
        max_blocks: 只统计前 N 块，仅用于自检。

    Returns:
        ``(ppl, n_tokens)``：PPL 为浮点数，``n_tokens`` 为参与统计的
        有效 token 总数。

    Raises:
        ValueError: 有效 token 数为 0，即 ``len(ids)`` 不足以覆盖两个块。

    Note:
        交叉熵以 ``reduction="sum"`` 累计后再除以总 token 数，即宏平均；
        logits 须先转 ``float32``，bf16 精度不足以稳定求和。
        循环上界取 ``len(ids) - chunk``，以保证每个块都是完整长度。
    """
    dev = next(model.parameters()).device
    total_nll = 0.0
    total_tok = 0
    n_blocks = 0
    for i in range(0, len(ids) - chunk, chunk):
        if max_blocks is not None and n_blocks >= max_blocks:
            break

        inp = ids[i:i + chunk].unsqueeze(0).to(dev)      # [1, chunk]

        with torch.no_grad():
            logits = model(inp).logits[0]                    # [chunk, vocab]

        if shift:
            # 位置 0..T-2 的 logits 预测位置 1..T-1 的真实 token。
            sl = logits[:-1, :]                              # [chunk-1, vocab]
            lb = inp[0, 1:]                                  # [chunk-1]

            # NOTE: 该断言只能在 shift=True 分支内成立 —— shift=False 时
            #   logits 与 label 均为 chunk 个元素，配对失衡，
            #   故断言必须放在分支内而非循环外。
            assert sl.shape[0] == lb.shape[0] == inp.shape[1] - 1, (
                f"shift 写错了！logits={sl.shape[0]} labels={lb.shape[0]} "
                f"expected={inp.shape[1] - 1}")
        else:
            # NOTE: 不做位置右移，仅供 --selftest 展示错位时的 PPL 偏差。
            sl = logits
            lb = inp[0]

        total_nll += F.cross_entropy(sl.float(), lb, reduction="sum").item()
        total_tok += lb.numel()
        n_blocks += 1

    if total_tok == 0:
        raise ValueError(f"序列太短：len(ids)={len(ids)}, chunk={chunk}，"
                         f"至少要 {chunk * 2} 个 token")

    # PPL = exp(交叉熵) = exp(总 nll / 总 token 数)
    return math.exp(total_nll / total_tok), total_tok


def evaluate(model, tok, chunk: int = CHUNK) -> dict:
    out = {}
    for lang in ("en", "zh"):
        ids = tok(load_text(lang), return_tensors="pt").input_ids[0]
        p, n = ppl_from_ids(model, ids, chunk=chunk)
        out[lang] = {"ppl": round(p, 4), "tokens": n}
    return out



def build(kind: str, name: str = "Qwen/Qwen2.5-1.5B-Instruct"):
    from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig)

    import logging
    logging.getLogger("bitsandbytes").setLevel(logging.ERROR)

    tok = AutoTokenizer.from_pretrained(name)

    if kind == "bf16":
        m = AutoModelForCausalLM.from_pretrained(
            name, device_map={"": 0}, **_dtype_kwarg(torch.bfloat16))
    elif kind == "int8":
        m = AutoModelForCausalLM.from_pretrained(
            name, device_map={"": 0},
            quantization_config=BitsAndBytesConfig(load_in_8bit=True))
    elif kind == "int4":
        m = AutoModelForCausalLM.from_pretrained(
            name, device_map={"": 0},
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True))
    else:
        raise ValueError(kind)
    return m.eval(), tok


def run_m1(name: str = "Qwen/Qwen2.5-1.5B-Instruct",
           with_speed: bool = False) -> list[dict]:
    """依次评测 bf16 / int8 / int4 三配置，返回对比表原始数据。

    Args:
        name: Hugging Face 模型名。
        with_speed: 为 ``True`` 时额外测量 TTFT / TPOT；测量前先校验 GPU
            状态，避免在降频状态下取数。

    Returns:
        每配置一条记录，字段为 ``config`` / ``weight_gib``（单位 GiB）/
        ``ppl_en`` / ``ppl_zh`` / ``tok_en`` / ``tok_zh``；``with_speed``
        为真时追加 ``ttft_ms`` / ``tpot_ms`` / ``decode_tps`` /
        ``peak_gib``（单位 GiB）。

    Raises:
        RuntimeError: ``with_speed`` 为真且 GPU 未就绪（如已降频）。

    Note:
        ``weight_gib`` 在加载完成、首次前向之前读取，此时显存中只有权重。
    """
    rows = []
    for kind in ("bf16", "int8", "int4"):
        import gc
        torch.cuda.empty_cache()
        m, tok = build(kind, name)
        # 在首次前向之前读取显存，此时只有权重占用。
        weight_gib = torch.cuda.memory_allocated() / 1024 ** 3
        res = evaluate(m, tok)
        row = {
            "config": kind,
            "weight_gib": round(weight_gib, 4),
            "ppl_en": res["en"]["ppl"],
            "ppl_zh": res["zh"]["ppl"],
            "tok_en": res["en"]["tokens"],
            "tok_zh": res["zh"]["tokens"],
        }
        if with_speed:
            from aq.bench import bench_model, check_gpu_ready
            check_gpu_ready()          # GPU 降频时直接抛错，不产出无效数据
            b = bench_model(m, tok, prompt_len=128, max_new_tokens=32)
            row.update({"ttft_ms": b["ttft_ms"], "tpot_ms": b["tpot_ms"],
                        "decode_tps": b["decode_tps"], "peak_gib": b["peak_gib"]})
        rows.append(row)
        del m, tok
        gc.collect()
        torch.cuda.empty_cache()
    return rows


def print_m1_table(rows: list[dict]) -> str:
    """把 :func:`run_m1` 的结果渲染成 markdown 表格。

    Args:
        rows: :func:`run_m1` 的返回值，首条记录作为 bf16 基线。

    Returns:
        渲染出的 markdown 表格文本，同时打印到标准输出。

    Note:
        ``Δen`` / ``Δzh`` 为相对首行基线的百分比变化；末尾另打印
        理论显存（全部参数量 × bits/8）与实测值的比值。
    """
    base = rows[0]
    has_speed = "ttft_ms" in base

    head = "| 配置 | 权重 GiB | 占 bf16 | PPL_en | Δen | PPL_zh | Δzh |"
    sep = "|---|---|---|---|---|---|---|"
    if has_speed:
        head += " TTFT ms | TPOT ms | tok/s |"
        sep += "---|---|---|"

    lines = [head, sep]
    for r in rows:
        wr = r["weight_gib"] / base["weight_gib"] * 100
        den = (r["ppl_en"] - base["ppl_en"]) / base["ppl_en"] * 100
        dzh = (r["ppl_zh"] - base["ppl_zh"]) / base["ppl_zh"] * 100
        d_en = "—" if r is base else f"{den:+.2f}%"
        d_zh = "—" if r is base else f"{dzh:+.2f}%"
        line = (f"| **{r['config']}** | {r['weight_gib']:.4f} | {wr:.1f}% "
                f"| {r['ppl_en']:.4f} | {d_en} | {r['ppl_zh']:.4f} | {d_zh} |")
        if has_speed:
            line += (f" {r['ttft_ms']:.2f} | {r['tpot_ms']:.2f} "
                     f"| {r['decode_tps']:.1f} |")
        lines.append(line)

    out = "\n".join(lines)
    print("\n" + out)

    # 理论显存与实测值对比，单位 GiB。
    print("\n理论 vs 实测（理论 = 全部参数 × bits/8）")
    N = 1_543_714_304
    for r, bits in zip(rows, (16, 8, 4)):
        theo = N * bits / 8 / 1024 ** 3
        print(f"  {r['config']:5} 理论 {theo:6.4f} GiB  实测 {r['weight_gib']:6.4f} GiB  "
              f"比值 {r['weight_gib']/theo:.2f}")

    return out
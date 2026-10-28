"""速度基准测试：TTFT / TPOT / 端到端吞吐。

指标口径：

- TTFT：首 token 延迟，对应 prefill 阶段；算力受限，随 prompt 长度先次线性、
  后超线性增长。
- TPOT：每输出 token 时间，对应 decode 阶段；受内存带宽限制。
- 吞吐：总 token/s，随并发数上升（continuous batching）。
- 实测差距：同机 prefill 3799 token/s 对 decode 21.3 token/s，相差约 178 倍。

计时纪律，缺任一项则结果不可信：

- 每次测量前后都调用 ``torch.cuda.synchronize()``。CUDA 为异步执行，不 sync
  测到的是下发时间；大 matmul 场景实测低估 48.9 倍。
- 测量前先 warmup 3 次。首次迭代含 cuDNN 算法选择、kernel 加载与显存池分配，
  实测首次耗时为稳态的 13.6 倍。
- 报告均值与波动，不报告单次值。
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

import torch

# ============================================================================
# 常量（与 protocol.md 保持一致；改动此处须同步修改协议）
# ============================================================================

WARMUP: int = 3
REPEATS: int = 5


# ============================================================================
# 测量前置检查
# ============================================================================
BASELINE_PATH = ".gpu_baseline.json"
"""性能基线文件路径，内容为见过的最高 TFLOPS。"""


def _gpu_info() -> dict:
    """读取功耗、时钟与温度。

    这些数值只用于展示，不参与降频判定；判定依据见
    :func:`_micro_bench_tflops`。

    Returns:
        字典，键为 ``clock`` / ``clock_max``（MHz）、``power``（W）、
        ``temp``（摄氏度）。查询失败时返回空字典。
    """
    import subprocess
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=clocks.sm,clocks.max.sm,power.draw,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True).stdout.strip()
        cur, mx, pw, tp = (float(x) for x in out.split(", "))
        return {"clock": cur, "clock_max": mx, "power": pw, "temp": tp}
    except Exception:
        return {}


def _micro_bench_tflops(rounds: int = 3) -> float:
    """用固定负载测 bf16 GEMM 的 TFLOPS，多轮取中位数。

    降频判定用实测性能而非时钟频率，原因：

    1. ``clocks.max.sm`` 是硅片上限，笔记本受功耗墙限制永远达不到
       （测试机上报 3090 MHz，实际运行区间 1800–2030 MHz）。
    2. 空闲状态下读到的时钟明显偏低，误报率高。
    3. 需要反映的是实际算力，TFLOPS 直接给出该口径。

    Args:
        rounds: 采样轮数，最终取各轮结果的中位数。

    Returns:
        bf16 GEMM 的算力，单位 TFLOP/s。
    """
    import statistics

    import torch

    n, reps = 4096, 8
    a = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
    for _ in range(3):                      # warmup
        a @ b
    torch.cuda.synchronize()

    samples = []
    for _ in range(rounds):
        t0 = time.perf_counter()
        for _ in range(reps):
            a @ b
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / reps
        samples.append(2 * n ** 3 / dt / 1e12)      # TFLOPS

    return statistics.median(samples)


def check_gpu_ready(tol: float = 0.40, strict: bool = False,
                    baseline_path: str = BASELINE_PATH) -> dict:
    """在测量速度之前调用，与历史基线比较并在明显变慢时告警。

    自校准：首次运行记录基线，之后与基线比较，本次更快则抬高基线，
    避免首次在降频状态下运行导致基线偏低。

    Args:
        tol: 允许的相对跌幅。默认 0.40，因测试机功耗墙仅 40 W，
            短负载抖动可达 ±35%，故放宽阈值且只告警。
        strict: 为 True 时跌幅超限抛出异常，而非仅打印告警。
        baseline_path: 基线文件路径，默认为 ``BASELINE_PATH``。

    Returns:
        字典，键为 ``tflops`` / ``baseline`` / ``ratio`` /
        ``clock`` / ``power`` / ``temp``。

    Raises:
        RuntimeError: ``strict=True`` 且实测 TFLOPS 低于基线超过 ``tol``。

    Note:
        默认不阻断：绝对性能受功耗墙影响，但同机内 bf16 / INT8 / INT4 的
        相对比较依然有效，后者才是基准的用途。正式出报告时可传
        ``strict=True, tol=0.15`` 严格把关。
    """
    import json

    tf = _micro_bench_tflops()
    info = _gpu_info()
    path = Path(baseline_path)

    best = 0.0
    if path.exists():
        try:
            best = float(json.loads(path.read_text(encoding="utf-8"))["tflops"])
        except Exception:
            best = 0.0

    if best <= 0:
        path.write_text(json.dumps({"tflops": round(tf, 3)},
                                   ensure_ascii=False), encoding="utf-8")
        print(f"  [GPU] 已记录性能基线：{tf:.1f} TFLOPS（{path}）")
        return {"tflops": tf, "baseline": tf, "ratio": 1.0, **info}

    ratio = tf / best
    if tf > best:
        path.write_text(json.dumps({"tflops": round(tf, 3)},
                                   ensure_ascii=False), encoding="utf-8")

    pw = info.get("power", 0)
    tp = info.get("temp", 0)
    ck = info.get("clock", 0)
    print(f"  [GPU] {tf:.1f} TFLOPS / 基线 {best:.1f} = {ratio:.0%}"
          f"   功耗 {pw:.0f}W  温度 {tp:.0f}C  时钟 {ck:.0f} MHz")

    if ratio < 1 - tol:
        msg = (
            f"GPU 性能比历史基线低 {1 - ratio:.0%}：{tf:.1f} / {best:.1f} TFLOPS\n"
            f"  功耗 {pw:.0f}W，温度 {tp:.0f}C\n"
            f"  检查：1) 电源计划高性能  2) OEM 工具是否静音/省电\n"
            f"        3) 关闭浏览器硬件加速  4) 是否插电\n"
            f"  注意：精度和显存【不受影响】，只有速度数据受影响。\n"
            f"  若首次就在低性能状态跑，删掉 {path} 重新记录基线。"
        )
        if strict:
            raise RuntimeError(msg)
        print(f"  [WARN] {msg}")

    return {"tflops": tf, "baseline": best, "ratio": ratio, **info}

def measure(fn: Callable[[], None], warmup: int = WARMUP,
            repeats: int = REPEATS, sync: bool = True) -> tuple[float, float]:
    """测量 ``fn`` 的耗时。

    Args:
        fn: 无参可调用对象，内部负责向 GPU 提交工作。
        warmup: 预热次数，不计入统计。
        repeats: 记录次数。
        sync: 是否在每次调用前后执行 ``torch.cuda.synchronize()``。
            置为 False 用于对照实验，可复现不 sync 时低估耗时的现象。

    Returns:
        ``(均值, 半极差)``，单位秒。半极差为 ``(max - min) / 2``。

    Note:
        记录流程为 sync → t0 → fn() → sync → t1，预热阶段同样 sync。
        采用半极差而非标准差，是因为样本量通常只有 5 个，标准差估计不稳定。
    """
    for _ in range(warmup):
        fn()
        if sync :
            torch.cuda.synchronize()

    ts = []

    for _ in range(repeats):
        if sync:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if sync:
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        ts.append(t1 - t0)

    mean = sum(ts) / len(ts)
    spread = (max(ts) - min(ts)) / 2
    return mean, spread



def _dev(model):
    return next(model.parameters()).device

def _make_prompt(tok, n_tok: int,device):
    base = ("Machine learning is a field of study in artificial intelligence "
            "concerned with the statistical algorithms that computer systems "
            "use to learn from data and generalize to unseen data. ")
    ids = tok(base * 200, return_tensors="pt").input_ids[0][:n_tok]
    return ids.unsqueeze(0).to(device)    



def measure_ttft(model, tok, prompt_ids: torch.Tensor) -> tuple[float, float]:
    """测量 TTFT，即从提交到第一个 token 可用。

    实现为单次 forward，等价于 ``generate(max_new_tokens=1)`` 的首步耗时。

    Args:
        model: 已加载的因果语言模型。
        tok: 分词器，本函数不使用，保留以统一调用签名。
        prompt_ids: prompt 的 token id，一维或二维张量。

    Returns:
        ``(秒, 半极差)``。
    """
    d = _dev(model)
    # 1-D 输入补 batch 维；2-D 直接送。注意 dim 是方法，必须带括号调用。
    ids = prompt_ids.unsqueeze(0).to(d) if prompt_ids.dim() == 1 else prompt_ids.to(d)

    def one():
        with torch.no_grad():
            model(ids)

    return measure(one)

def measure_decode(model, tok, prompt_ids: torch.Tensor,
                   n_tokens: int = 32) -> tuple[float, float]:
    """测量 decode 阶段的单 token 耗时（TPOT）。

    Note:
        必须使用 ``past_key_values``（KV cache）。省略 cache 时测到的是
        “每次重算整个前缀”的耗时，数值偏大且随前缀长度线性增长。

    Args:
        model: 已加载的因果语言模型。
        tok: 分词器，本函数不使用，保留以统一调用签名。
        prompt_ids: prompt 的 token id，一维或二维张量。
        n_tokens: 记录的 decode 步数，预热步数另计。

    Returns:
        ``(每 token 秒数, 半极差)``。
    """
    d = _dev(model)
    # 1-D 输入补 batch 维；2-D 直接送。注意 dim 是方法，必须带括号调用。
    ids = prompt_ids.unsqueeze(0).to(d) if prompt_ids.dim() == 1 else prompt_ids.to(d)

    with torch.no_grad():
        out = model(ids, use_cache=True)
    past = out.past_key_values
    nxt = out.logits[:, -1:].argmax(-1)

    ts = []
    for i in range(WARMUP + n_tokens):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            o = model(nxt, past_key_values=past, use_cache=True)
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
        past = o.past_key_values
        nxt = o.logits[:, -1:].argmax(-1)

    use = ts[WARMUP:]
    return sum(use) / len(use), (max(use) - min(use)) / 2


# ============================================================================
# 汇总
# ============================================================================
def bench_model(model, tok, prompt_len: int = 128, max_new_tokens: int = 64) -> dict:
    """跑完整一组速度基准，返回可写入 env.md / protocol.md 的字典。

    Args:
        model: 已加载的因果语言模型。
        tok: 分词器。
        prompt_len: 目标 prompt token 数，实际长度可能被文本量截断。
        max_new_tokens: 生成步数上限；实际生成数受 EOS 影响。

    Returns:
        字典，字段包括 ``ttft_ms`` / ``tpot_ms`` / ``decode_tps`` /
        ``prefill_tps`` / ``tps_e2e`` / ``peak_gib`` / ``prompt_len`` /
        ``max_new_tokens``，以及 ``n_generated`` / ``e2e_seconds`` 和各计时
        量的半极差字段（``*_spread_ms``）。时间单位为毫秒，``peak_gib``
        单位为 GiB，吞吐单位为 token/s。

    Raises:
        AssertionError: 峰值显存超过 8 GB 卡容量的 90%。
    """
    d = _dev(model)
    ids = _make_prompt(tok, prompt_len, d)
    real_len = int(ids.shape[1])

    ttft, ttft_sp = measure_ttft(model, tok, ids)
    tpot, tpot_sp = measure_decode(model, tok, ids, n_tokens=max_new_tokens)

    # NOTE: 先跑一次取实际生成的 token 数，模型可能提前触发 EOS。
    with torch.no_grad():
        g = model.generate(ids, max_new_tokens=max_new_tokens,
                           do_sample=False, pad_token_id=tok.eos_token_id)
    n_new = int(g.shape[1] - ids.shape[1])

    def gen():
        with torch.no_grad():
            model.generate(ids, max_new_tokens=max_new_tokens,
                           do_sample=False, pad_token_id=tok.eos_token_id)

    e2e, _ = measure(gen, warmup=1, repeats=3)

    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        model(ids)
    peak = torch.cuda.max_memory_allocated() / 1024 ** 3

    # NOTE: 8 GB 卡的边界条件：显存溢出到共享内存不报错，只是静默变慢。
    assert peak < 8 * 0.9, f"峰值 {peak:.2f} GiB 接近 8 GB，可能已溢出"

    return {
        "prompt_len": real_len,
        "max_new_tokens": max_new_tokens,
        "n_generated": n_new,
        "ttft_ms": round(ttft * 1000, 2),
        "ttft_spread_ms": round(ttft_sp * 1000, 2),
        "tpot_ms": round(tpot * 1000, 2),
        "tpot_spread_ms": round(tpot_sp * 1000, 2),
        "decode_tps": round(1 / tpot, 1),
        "prefill_tps": round(real_len / ttft, 1),
        "e2e_seconds": round(e2e, 3),
        # 吞吐口径：只统计生成的 token，不含 prompt。
        "tps_e2e": round(n_new / e2e, 1),
        "peak_gib": round(peak, 3),
    }


def sweep_prompt_len(model, tok, lengths: tuple[int, ...] = (128, 256, 512, 1024, 2048)) -> list[dict]:
    """按给定长度扫描 prompt，定位 TTFT 增长由次线性转为超线性的拐点。

    Args:
        model: 已加载的因果语言模型。
        tok: 分词器。
        lengths: 待扫描的 prompt token 数序列。

    Returns:
        列表，每项含 ``prompt_len`` / ``ttft_ms`` / ``spread_ms``。
    """
    rows = []
    for n in lengths:
        ids = _make_prompt(tok, n, _dev(model))
        ttft, sp = measure_ttft(model, tok, ids)
        rows.append({"prompt_len": int(ids.shape[1]),
                     "ttft_ms": round(ttft * 1000, 2),
                     "spread_ms": round(sp * 1000, 2)})
    return rows

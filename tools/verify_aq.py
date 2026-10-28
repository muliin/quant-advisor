"""
verify_aq.py — 检验 aq/bench.py 与 aq/metrics.py 写得对不对

用法：
    python verify_aq.py --fast     # 只跑模型无关的行为自检（约 3 秒）
    python verify_aq.py            # 全量：+ 加载模型做官方对账 + 已知答案对照（约 1 分钟）

三层检验（沿用 10.15 的两道防线，再加一层"已知答案"）：
  ① 自检     = 内部一致性（不变量必须成立）
  ② 对账     = 与权威来源比（官方 loss）
  ③ 已知答案 = 与 10.19 实测基线对齐

退出码：0 = 全过；1 = 有失败；2 = 环境问题（如无 aq 包）
"""
from __future__ import annotations

# 允许从任意目录运行：把项目根目录加入 sys.path
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import argparse
import math
import sys
import time

PASS, FAIL, SKIP = 0, [], []


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS
    if cond:
        PASS += 1
        print(f"  [OK]   {name:<46} {detail}")
    else:
        FAIL.append((name, detail))
        print(f"  [FAIL] {name:<46} {detail}")


def skip(name: str, why: str) -> None:
    SKIP.append((name, why))
    print(f"  [SKIP] {name:<46} {why}")


def section(t: str) -> None:
    print(f"\n{'=' * 72}\n{t}\n{'=' * 72}")


# ============================================================================
# ① 行为自检：measure() 的 warmup / sync / 半极差是否真的生效
# ============================================================================
class SlowFirst:
    """第一次很慢、之后很快的可控函数 —— 用来验证 warmup 真的跳过了前几次"""

    def __init__(self, first: float = 0.08, rest: float = 0.004):
        self.n, self.first, self.rest = 0, first, rest

    def __call__(self):
        self.n += 1
        time.sleep(self.first if self.n == 1 else self.rest)


def t_measure_behaviour():
    section("① 行为自检 —— measure()（不需要模型）")
    from aq.bench import measure

    # --- T1: 返回结构 ---
    ret = measure(lambda: time.sleep(0.002))
    check("measure() 返回二元组", isinstance(ret, tuple) and len(ret) == 2,
          f"{type(ret).__name__} len={len(ret) if isinstance(ret, tuple) else '-'}")
    mean, spread = ret
    check("两个分量都是 float", isinstance(mean, float) and isinstance(spread, float),
          f"mean={type(mean).__name__} spread={type(spread).__name__}")
    check("均值 > 0 且波动 >= 0", mean > 0 and spread >= 0,
          f"mean={mean*1000:.2f}ms spread={spread*1000:.2f}ms")
    check("波动的量级不荒唐（spread < mean）", spread < mean,
          f"spread/mean={spread/mean:.2f}")

    # --- T2: ★★ warmup 真的跳过了前几次 ---
    # ★ 用 repeats=1 才无歧义：warmup=0 时唯一一次就是"慢"的那次
    fn0 = SlowFirst()
    m0, _ = measure(fn0, warmup=0, repeats=1, sync=False)
    check("warmup=0 时测得的就是第一次的慢（≈80ms）", m0 > 0.05,
          f"mean={m0*1000:.1f}ms（第一睡 80ms）")

    fn3 = SlowFirst()
    m3, _ = measure(fn3, warmup=3, repeats=3, sync=False)
    check("warmup=3 时慢的那次被跳过", m3 < 0.02,
          f"mean={m3*1000:.1f}ms")
    check("warmup 使结果至少小 3 倍", m0 / m3 > 3,
          f"{m0/m3:.1f} 倍")

    # --- T3: warmup 真的调用了，不是被忽略 ---
    fn = SlowFirst()
    measure(fn, warmup=5, repeats=2, sync=False)
    check("warmup=5 + repeats=2 共调用 7 次", fn.n == 7, f"实际 {fn.n} 次")

    # --- T4: repeats 生效 ---
    fn = SlowFirst()
    measure(fn, warmup=0, repeats=9, sync=False)
    check("repeats=9 共调用 9 次", fn.n == 9, f"实际 {fn.n} 次")


def t_measure_sync():
    section("① 行为自检 —— measure() 的 sync 开关（需要 GPU）")
    import torch
    if not torch.cuda.is_available():
        skip("sync 对照", "无 CUDA")
        return
    from aq.bench import measure

    a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    m_sync, _ = measure(lambda: a @ a, sync=True)
    m_no, _ = measure(lambda: a @ a, sync=False)
    check("sync=False 明显偏小（测的是下发时间）", m_no < m_sync / 2,
          f"sync={m_sync*1000:.3f}ms  no-sync={m_no*1000:.3f}ms  ({m_sync/m_no:.0f}×)")
    check("sync=True 测得的时间与 10.18 量级一致（3–15 ms）",
          0.003 < m_sync < 0.015, f"{m_sync*1000:.2f} ms")


# ============================================================================
# ② PPL：结构自检（不需要模型）
# ============================================================================
def t_ppl_structure():
    section("② 结构自检 —— metrics.py（不需要模型）")
    import aq.metrics as M

    for fn in ("ppl_from_ids", "load_text"):
        check(f"存在 {fn}()", callable(getattr(M, fn, None)))

    for lang in ("en", "zh"):
        try:
            t = M.load_text(lang)
            check(f"load_text({lang!r}) 可读", len(t) > 1500, f"{len(t)} 字符")
        except Exception as e:
            check(f"load_text({lang!r}) 可读", False, f"{type(e).__name__}: {e}")

    # --- chunk 太短应报错 ---
    import torch

    class Dummy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.p = torch.nn.Parameter(torch.zeros(1))

    try:
        M.ppl_from_ids(Dummy(), torch.zeros(10, dtype=torch.long), chunk=512)
        check("序列短于 chunk 时应报错", False, "没有抛异常")
    except (ValueError, RuntimeError) as e:
        check("序列短于 chunk 时应报错", True, f"{type(e).__name__}")
    except Exception as e:
        check("序列短于 chunk 时应报错", False, f"抛了意外类型 {type(e).__name__}")


# ============================================================================
# ③ 对账 + 已知答案（需要模型）
# ============================================================================
# ★ 这些数字来自 2026-10-19 的实测（见 10.19 教材 §6.1）
KNOWN = {
    "bf16": {"ppl_en": 13.1653, "ppl_zh": 11.6734, "weight_gib": 2.8754},
    "int8": {"ppl_en": 13.2480, "ppl_zh": 11.8813, "weight_gib": 1.6665},
    "int4": {"ppl_en": 14.0815, "ppl_zh": 13.2578, "weight_gib": 1.0832},
}
CHUNK = 512
TOL_PPL = 0.02          # ±2%：容忍浮点累加顺序与版本差异
TOL_MEM = 0.10          # ±10%：容忍版本差异（10.19 发现云端/本地差过 1 GiB）
KNOWN_TPOT_MS = 46.90   # 10-18 本地实测的 bf16 TPOT


def t_ppl_reconcile(model, tok):
    section("③ 官方对账 —— ppl_from_ids vs model(labels=...)")
    import torch
    import aq.metrics as M

    ids = tok(M.load_text("en"), return_tensors="pt").input_ids[0]
    dev = next(model.parameters()).device

    # --- R1: 第一块 vs 官方 loss ---
    inp = ids[:CHUNK].unsqueeze(0).to(dev)
    with torch.no_grad():
        official = math.exp(model(inp, labels=inp).loss.item())
    mine, _ = M.ppl_from_ids(model, ids, chunk=CHUNK, shift=True, max_blocks=1)
    check("与官方 labels 捷径等价（差 < 1e-5）", abs(official - mine) < 1e-5,
          f"官方={official:.6f} 我的={mine:.6f} 差={abs(official-mine):.2e}")

    # --- R2: 不 shift 必须暴涨 ---
    bad, _ = M.ppl_from_ids(model, ids, chunk=CHUNK, shift=False, max_blocks=1)
    check("不 shift 时 PPL 暴涨（> 1e4）", bad > 1e4,
          f"不shift={bad:,.1f} vs 正确={mine:.4f}")

    # --- R3: 确定性 ---
    p1, _ = M.ppl_from_ids(model, ids, chunk=CHUNK, max_blocks=2)
    p2, _ = M.ppl_from_ids(model, ids, chunk=CHUNK, max_blocks=2)
    check("重复两次逐位相同", p1 == p2, f"{p1:.12f}")


def t_known_answers(model, tok):
    section("③ 已知答案对照 —— 与 10.19 实测基线比")
    import torch
    import aq.metrics as M

    for lang in ("en", "zh"):
        ids = tok(M.load_text(lang), return_tensors="pt").input_ids[0]
        p, n = M.ppl_from_ids(model, ids, chunk=CHUNK)
        want = KNOWN["bf16"][f"ppl_{lang}"]
        rel = abs(p - want) / want
        check(f"bf16 PPL_{lang} 与基线一致（±{TOL_PPL:.0%}）", rel < TOL_PPL,
              f"我的={p:.4f} 基线={want:.4f} 偏差={rel:.2%}")

    w = torch.cuda.memory_allocated() / 1024 ** 3
    want_w = KNOWN["bf16"]["weight_gib"]
    rel = abs(w - want_w) / want_w
    check(f"bf16 权重显存与基线一致（±{TOL_MEM:.0%}）", rel < TOL_MEM,
          f"我的={w:.4f} GiB 基线={want_w:.4f} 偏差={rel:.2%}")


def t_bench_model(model, tok):
    section("③ bench_model() 的结构与量级")
    import aq.bench as B

    r = B.bench_model(model, tok, prompt_len=128, max_new_tokens=32)
    need = {"prompt_len", "ttft_ms", "tpot_ms", "decode_tps",
            "prefill_tps", "peak_gib"}
    missing = need - set(r)
    check("返回字典含必需字段", not missing,
          f"缺 {missing}" if missing else f"{len(r)} 个字段")

    check("TTFT 在合理量级（20–400 ms）", 20 < r.get("ttft_ms", -1) < 400,
          f"{r.get('ttft_ms')} ms")
    check("TPOT 在合理量级（10–400 ms）", 10 < r.get("tpot_ms", -1) < 400,
          f"{r.get('tpot_ms')} ms")
    check("峰值显存 < 3.2 GiB（bf16 1.5B）", 2.5 < r.get("peak_gib", 0) < 3.2,
          f"{r.get('peak_gib')} GiB")

    # ★ 用【性能】判断是否降频，比读时钟可靠
    #   原因：nvidia-smi 在 GPU 空闲时读到的时钟本来就会掉，会误报
    tpot = r.get("tpot_ms", 0)
    if tpot > KNOWN_TPOT_MS * 1.5:
        print(f"  [WARN] TPOT {tpot:.1f} ms 比基线 {KNOWN_TPOT_MS} ms 慢 "
              f"{tpot/KNOWN_TPOT_MS:.1f} 倍 —— 很可能在降频，速度数据不可信"
              f"（见 10.19 §6.1.1）")
    elif tpot < KNOWN_TPOT_MS * 0.5:
        print(f"  [WARN] TPOT {tpot:.1f} ms 比基线快 {KNOWN_TPOT_MS/tpot:.1f} 倍 "
              f"—— 检查是否测的是不同精度/不同模型")
    else:
        print(f"  [INFO] TPOT {tpot:.1f} ms 与基线 {KNOWN_TPOT_MS} ms 同量级 "
              f"（{tpot/KNOWN_TPOT_MS:.2f}×）→ 测量可信")

    # 读一次实时功耗/温度作为参考（不算判据）
    try:
        import subprocess
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=power.draw,temperature.gpu,clocks.sm,clocks.max.sm",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True).stdout.strip()
        pw, tp, cur, mx = (float(x) for x in out.split(", "))
        print(f"  [INFO] 空闲态：{pw:.0f}W {tp:.0f}°C {cur:.0f}/{mx:.0f} MHz"
              f"（空闲时钟会掉，不用它判降频）")
    except Exception:
        pass


# ============================================================================
# main
# ============================================================================
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="只跑模型无关的自检")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    args = ap.parse_args()

    print("verify_aq.py — 检验 aq/bench.py 与 aq/metrics.py")

    try:
        import aq.bench, aq.metrics  # noqa: F401
    except Exception as e:
        print(f"\n[环境] 无法导入 aq 包：{type(e).__name__}: {e}")
        print("      请在项目根目录运行，并确认 aq/ 目录存在。")
        return 2

    # ---- 模型无关部分 ----
    try:
        t_measure_behaviour()
    except NotImplementedError as e:
        check("aq.bench.measure() 已实现", False, f"NotImplementedError: {e}")
    except Exception as e:
        check("aq.bench.measure() 可运行", False, f"{type(e).__name__}: {e}")

    try:
        t_measure_sync()
    except Exception as e:
        check("measure() sync 对照", False, f"{type(e).__name__}: {e}")

    try:
        t_ppl_structure()
    except NotImplementedError as e:
        check("aq.metrics 已实现", False, f"NotImplementedError: {e}")
    except Exception as e:
        check("aq.metrics 结构自检", False, f"{type(e).__name__}: {e}")

    if not args.fast:
        section("加载模型（bf16）")
        import logging
        logging.getLogger("bitsandbytes").setLevel(logging.ERROR)
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from aq.metrics import _dtype_kwarg

        t0 = time.perf_counter()
        tok = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, device_map={"": 0},
            **_dtype_kwarg(torch.bfloat16)).eval()
        print(f"  加载耗时 {time.perf_counter()-t0:.1f}s")

        for fn in (t_ppl_reconcile, t_known_answers, t_bench_model):
            try:
                fn(model, tok)
            except NotImplementedError as e:
                check(fn.__name__, False, f"NotImplementedError: {e}")
            except Exception as e:
                check(fn.__name__, False, f"{type(e).__name__}: {e}")

    # ---- 汇总 ----
    print(f"\n{'=' * 72}")
    if SKIP:
        print(f"跳过 {len(SKIP)} 项：")
        for n, w in SKIP:
            print(f"  - {n}: {w}")
    total = PASS + len(FAIL)
    if not FAIL:
        print(f"全部通过：{PASS}/{total}"
              + ("（fast 模式，未做模型对账）" if args.fast else ""))
        print("=" * 72)
        return 0
    print(f"通过 {PASS}/{total}，失败 {len(FAIL)} 项：")
    for n, d in FAIL:
        print(f"  ✗ {n}   {d}")
    print("=" * 72)
    return 1


if __name__ == "__main__":
    sys.exit(main())

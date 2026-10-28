"""量化顾问 Agent。

两个正交的维度决定方案：重要性信号 × 候选位宽，而非并列的"策略按钮"。
由此产生四条约束：

- 始终带基线：每个方案同时给出"无信号（均匀）"基线做对照
- 实测 / 估计必须标注：有真数据标"实测"，否则标"估计"并说明依据
- 自我警告：加法预测会系统性低估约 2 倍（实测），必须主动提示
- 规则路由：不依赖任何 LLM，未配置 key 或断网时仍可运行

数据来源：

- ``dense``：``results/sensitivity_en_nf4.csv``（196 模块 × {2,4,8}）
- ``moe``：``results/sensitivity_moe.csv``（2048 专家 × {2}）
- ``moe`` 表：``results/bit_layer_full.csv``（16 层 × {2,3,4,5,6,8}），
  MoE 方案使用此项
"""
from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

GIB = 1024 ** 3

try:
    from aq.models import describe_model as _describe_model
    from aq.models import match_model as _match_model
    _MODELS = True
except Exception:
    _MODELS = False

try:
    from aq.llm_parse import available as _llm_available
    from aq.llm_parse import llm_parse as _llm_parse
    _LLM = True
except Exception:
    _LLM = False

try:
    from aq.knowledge import cite as _cite
    from aq.knowledge import conflicts_block as _conflicts_block
    from aq.knowledge import query as _kb_query
    from aq.knowledge import stats as _kb_stats
    _KB = True
except Exception:                                   # 知识库缺失时静默降级
    _KB = False

__all__ = ["Query", "parse_query", "answer", "TOOLS", "SYSTEM_CARD",
           "DATASETS", "available_datasets"]

# ============================================================ 数据注册表
DATASETS = {
    "dense": {
        "path": "results/sensitivity_en_nf4.csv",
        "label": "逐层 / 逐模块",
        "unit": "个 Linear",
        "model": "Qwen2.5-1.5B（dense）",
        "bits": (2, 4, 8),
        "n": 196,
    },
    "moe": {
        "path": "results/sensitivity_moe.csv",
        "label": "逐专家",
        "unit": "个专家决策点",
        "model": "OLMoE-1B-7B（16 层 × 64 专家 × 2 张量）",
        "bits": (2,),
        "n": 2048,
    },
}

BIT_TABLE = "results/bit_layer_full.csv"

# 实测参照点（OLMoE 逐层，平均位宽 → 实测 ΔPPL%）
# 来源：results/MoE逐专家扫描-实测结果.md
MEASURED_REF = {
    "moe": {
        2.0: 1902.500,   # 全 2bit，严重退化
        3.0: 15.803,     # 全 3bit
        3.5: 7.396,      # 8×3bit + 8×4bit（方案 A）
        4.0: 4.733,      # 全 4bit
    }
}

# 加法模型偏差不是常数：
# 均匀 4bit：预测 +4.30%，实测 +4.733%（1.10×）
# DP 择优选出的方案：预测 +3.50%，实测 +7.21%（2.06×）
# 优化器越挑，低估越严重，因此给区间而不是单点。
BIAS_RANGE = (1.10, 2.06)

# ============================================================ 意图解析
_MODEL_RE = re.compile(r"(\d+(?:\.\d+)?)\s*[Bb](?![A-Za-z0-9])")
_VRAM_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:[Gg][Bb]?|[Gg]i[Bb]?)(?![A-Za-z0-9])")
_MOE_WORDS = ("moe", "专家", "olmoe", "路由", "router", "混合专家")
# HF 名称（org/name）或本地路径（含 \ 或 / 且有 .bin/.safetensors 目录特征）
_MODELREF_RE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.\-]*/[A-Za-z0-9_][A-Za-z0-9_.\-]*")
_PATH_RE = re.compile(r"[A-Za-z]:\\[^\s，,。]+|(?:/[\w.\-]+){2,}")


@dataclass
class Query:
    raw: str = ""
    intent: str = "fit"          # fit | sensitivity | plan | speed | table
    model_b: float | None = None
    vram_gib: float | None = None
    dataset: str = "dense"
    signal: str = "measured"     # uniform | alphaq | measured | hybrid
    cand: str = "348"            # 34 | 348 | 2348
    model_ref: str | None = None  # HF 名称或本地路径（onboard 用）
    known: dict | None = None      # match_model 的结果（含 total_b/active_b）
    inherited: list[str] = field(default_factory=list)  # 从上下文继承的字段
    notes: list[str] = field(default_factory=list)


def parse_query(text: str, use_llm: bool = True,
                context: str | None = None) -> Query:
    q = Query(raw=text)
    t = text.lower()

    # 自检优先级最高，否则 "llm 自检" 会被其他关键词先匹配
    if ("selftest" in t or "自检" in t or "自测" in t
            or ("llm" in t and "api" in t)):
        q.intent = "selftest"
        return q

    # 先尝试 LLM 解析；未配置 key、调用失败或超时则跳过，纯规则路径仍可用
    llm = None
    if use_llm and _LLM and _llm_available():
        llm = _llm_parse(text)

    # ---------- 规则先跑一遍（作为 baseline） ----------

    if any(k in t for k in ("量化我的", "帮我量化", "扫描我的", "跑一次扫描",
                              "我有一个模型", "onboard", "真扫", "实际扫")):
        q.intent = "onboard"
    elif any(k in t for k in ("初始化", "开始", "能用吗", "能不能测",
                                "评估一下", "init")):
        q.intent = "init"
    elif any(k in t for k in ("测我的", "我的设备", "测一下我的", "我的机器",
                                "测本机", "probe", "测速", "什么显卡")):
        q.intent = "probe"
    elif any(k in t for k in ("依据", "来源", "证据", "矛盾",
                                "可靠", "可信", "知识库", "你测过", "凭什么")):
        q.intent = "knowledge"
    elif any(k in t for k in ("表", "矩阵", "位宽×层", "每层", "位宽",
                              "跑多少")):
        q.intent = "table"
    elif any(k in t for k in ("敏感", "哪些层", "哪些专家", "不能压", "sensitive")):
        q.intent = "sensitivity"
    elif any(k in t for k in ("方案", "分配", "怎么压", "plan", "allocate",
                              "混合精度")):
        q.intent = "plan"
    elif any(k in t for k in ("多快", "速度", "吞吐", "tok/s", "speed", "延迟",
                              "vllm", "后端", "推理框架", "加速", "快多少")):
        q.intent = "speed"

    if any(k in t for k in _MOE_WORDS):
        q.dataset = "moe"

    if any(k in t for k in ("均匀", "baseline", "基线", "不分配")):
        q.signal = "uniform"
    elif any(k in t for k in ("alphaq", "谱", "免校准", "spectral")):
        q.signal = "alphaq"
    elif any(k in t for k in ("混合", "hybrid")):
        q.signal = "hybrid"

    if "2bit" in t or "2 bit" in t:
        q.cand = "2348"
        q.notes.append("⚠️ 允许 2bit —— 实测说这是悬崖，强烈不建议")

    mr = _MODELREF_RE.search(text) or _PATH_RE.search(text)
    if mr:
        q.model_ref = mr.group(0)

    # 已知模型表比正则可靠（"deepseekV3" → 671B MoE）
    if _MODELS:
        q.known = _match_model(text)
        if q.known:
            q.model_b = q.known["total_b"]
            q.bits_note = q.known["name"]

    m = _MODEL_RE.search(text)
    if m:
        q.model_b = float(m.group(1))
    cands = [float(x) for x in _VRAM_RE.findall(text)]
    if cands:
        q.vram_gib = max(cands)
    # 已知模型表的 total_b 覆盖正则推断值
    if q.known:
        q.model_b = q.known["total_b"]

    # ---------- LLM 结果覆盖（LLM 优先，但只填它有把握的字段） ----------
    if llm:
        if llm.get("intent"):
            q.intent = llm["intent"]
            q.notes.append("意图：LLM 解析")
        if llm.get("model_b"):
            q.model_b = float(llm["model_b"])
        if llm.get("vram_gib"):
            q.vram_gib = float(llm["vram_gib"])
        if llm.get("signal") in ("uniform", "alphaq", "measured", "hybrid"):
            q.signal = llm["signal"]
        if llm.get("cand") in ("34", "348", "2348"):
            q.cand = llm["cand"]
        if not q.known and _MODELS and llm.get("model_name"):
            q.known = _match_model(llm["model_name"])
            if q.known:
                q.model_b = q.known["total_b"]
        q.notes.append("数值仍全部来自实测")

    # "装不下 / 要多大卡"这类追问走 fit 解释，不走知识库
    if q.model_b is not None and any(
            k in text for k in ("装不下", "为什么", "多大卡", "需要多少",
                                "换卡", "换个")):
        q.intent = "fit"

    # ---------- 对话上下文继承（处理"那为什么装不下"这类追问） ----------
    if context:
        miss_model = q.model_b is None
        miss_vram = q.vram_gib is None
        miss_ref = q.model_ref is None
        if miss_model or miss_vram or miss_ref:
            prev = parse_query(context, use_llm=False)   # 只走规则，避免递归调 LLM
            if miss_model and prev.model_b is not None:
                q.model_b = prev.model_b
                q.known = q.known or prev.known
                q.inherited.append("模型规模")
            if miss_vram and prev.vram_gib is not None:
                q.vram_gib = prev.vram_gib
                q.inherited.append("显存")
            if miss_ref and prev.model_ref:
                q.model_ref = prev.model_ref
                q.inherited.append("模型名称")
            if q.inherited:
                q.notes.append("从上一句继承：" + "、".join(q.inherited))
    return q


# ============================================================ 工具
def _read_csv(path):
    p = Path(path)
    return list(csv.DictReader(open(p, encoding="utf-8"))) if p.exists() else None


def _size_gib(params: int, bits: int) -> float:
    eff = 4.128 if bits == 4 else float(bits)
    return params * eff / 8 / GIB


def tool_fit(model_b: float, vram_gib: float) -> dict:
    opts = []
    for bits in (16, 8, 4, 3):
        w = _size_gib(int(model_b * 1.029e9), bits)
        need = w + 0.5
        opts.append({"bits": bits, "weight_gib": round(w, 3),
                     "need_gib": round(need, 3),
                     "fits": need <= vram_gib * 0.8})
    return {"options": opts}


def tool_sensitivity(dataset: str = "dense") -> dict:
    meta = DATASETS.get(dataset, DATASETS["dense"])
    rows = _read_csv(meta["path"])
    if rows is None:
        return {"available": False, "path": meta["path"], "meta": meta}
    bits = meta["bits"][-1]
    sens = {r["target"]: float(r["rel_pct"])
            for r in rows if int(r["bits"]) == bits}
    order = sorted(sens, key=lambda k: -sens[k])
    vals = list(sens.values())
    return {"available": True, "meta": meta, "n": len(sens), "bits": bits,
            "top": [(k, sens[k]) for k in order[:5]],
            "worst": [(k, sens[k]) for k in order[-3:]],
            "max": max(vals), "min": min(vals),
            "spread": max(vals) - min(vals),
            "n_negative": sum(1 for v in vals if v < 0)}


def tool_bit_table() -> dict:
    rows = _read_csv(BIT_TABLE)
    if rows is None:
        return {"available": False}
    bits = [k for k in rows[0] if k != "layer"]
    tbl = {int(r["layer"]): {b: float(r[b]) for b in bits if r[b]}
           for r in rows}
    return {"available": True, "table": tbl, "bits": bits,
            "n_layers": len(tbl)}


MOE_LAYER_PARAMS = 64 * (2 * 1024 * 2048 + 2048 * 1024)


def _units_dense(cand: str):
    from aq.allocate import Unit
    rows = _read_csv(DATASETS["dense"]["path"])
    if rows is None:
        return None, None
    bits_ok = {"34": (4, 8), "348": (4, 8), "2348": (2, 4, 8)}.get(
        cand, (4, 8))
    by, params = {}, {}
    for r in rows:
        by.setdefault(r["target"], {})[int(r["bits"])] = float(r["rel_pct"])
        params[r["target"]] = int(r["params"])
    units = []
    for name, got in by.items():
        cost = {16: 0.0}
        for b in bits_ok:
            if b in got:
                cost[b] = max(0.0, math.log1p(got[b] / 100))
            else:
                near = min(got, key=lambda x: abs(x - b))
                cost[b] = max(0.0, math.log1p(got[near] / 100))
        units.append(Unit(name=name, params=params[name], cost=cost))
    return units, {"label": "逐模块", "n": len(units), "bits": bits_ok}


def _moe_cand_bits(cand: str):
    return {"34": (3, 4, 5, 6), "348": (3, 4, 5, 6),
            "2348": (2, 3, 4, 5, 6)}.get(cand, (3, 4, 5, 6))


def _units_moe(cand: str):
    from aq.allocate import Unit
    tt = tool_bit_table()
    if not tt.get("available"):
        return None, None
    bits_ok = _moe_cand_bits(cand)
    units = []
    for L, row in sorted(tt["table"].items()):
        cost = {16: 0.0}
        for b in bits_ok:
            if f"{b}bit" in row:
                cost[b] = max(0.0, math.log1p(row[f"{b}bit"] / 100))
        if len(cost) > 1:
            units.append(Unit(name=f"layer{L}", params=MOE_LAYER_PARAMS,
                              cost=cost))
    return units, {"label": "逐层（MoE）", "n": len(units), "bits": bits_ok}


def _units_eta(cand: str):
    from aq.allocate import Unit
    sp = _read_csv("results/alpha_moe.csv")
    if sp is None:
        return None, None
    bits_ok = _moe_cand_bits(cand)
    alphas = [float(r["alpha"]) for r in sp if r["alpha"]]
    abar = sorted(alphas)[len(alphas) // 2]
    va = sum((x - abar) ** 2 for x in alphas) / len(alphas)
    g = (min(alphas) * (max(alphas) - min(alphas)) / va) if va > 0 else 1.0
    per = {}
    for r in sp:
        per.setdefault(int(r["layer"]), []).append(
            (float(r["alpha"]), float(r["var"])))
    units = []
    for L, lst in sorted(per.items()):
        a_med = sorted(x[0] for x in lst)[len(lst) // 2]
        v_med = sorted(x[1] for x in lst)[len(lst) // 2]
        cost = {16: 0.0}
        for b in bits_ok:
            cost[b] = (abar / max(a_med, 1e-9)) ** g * v_med * (2.0 ** (-2 * b))
        units.append(Unit(name=f"layer{L}", params=MOE_LAYER_PARAMS, cost=cost))
    return units, {"label": "逐层（α 指标）", "n": len(units), "bits": bits_ok}


def tool_plan(dataset: str = "dense", signal: str = "measured",
              cand: str = "348", budget_gib: float | None = None) -> dict:
    """生成位宽分配方案，并附带均匀基线做对照。

    Args:
        dataset: 数据集名，取 ``dense`` 或 ``moe``。
        signal: 重要性信号，取 ``uniform`` / ``alphaq`` / ``measured`` /
            ``hybrid``。
        cand: 候选位宽集合标识，取 ``34`` / ``348`` / ``2348``。
        budget_gib: 显存预算，单位 GiB；``None`` 时按平均 3.5 bit 估算。

    Returns:
        字典，含 ``plan`` / ``hist`` / ``predicted_pct`` / ``uniform_predicted_pct``
        / ``avg_bits`` 等字段；``available`` 为 False 时 ``error`` 说明原因。
    """
    from aq.solve import solve_dp

    fixed = 0.434
    if dataset == "moe":
        if signal == "alphaq":
            units, info = _units_eta(cand)
        else:
            units, info = _units_moe(cand)
        uniform_bits = 4
    else:
        units, info = _units_dense(cand)
        uniform_bits = 4

    if not units:
        return {"available": False, "error": "缺少数据文件"}

    total = sum(u.params for u in units)
    if budget_gib is None:
        # 默认按平均 3.5 bit 给预算：均匀 4bit 预算下 DP 会退化成全 4bit
        budget_gib = fixed + total * 3.5 / 8 / GIB

    uni_cost = sum(u.cost.get(uniform_bits, 0.0) for u in units)

    if signal == "uniform":
        plan = {"assign": {u.name: uniform_bits for u in units},
                "cost": uni_cost, "gib": budget_gib}
    else:
        try:
            r = solve_dp(units, budget_gib, candidate_bits=info["bits"],
                         fixed_gib=fixed, n_buckets=4000)
        except ValueError as e:
            return {"available": False, "error": str(e)}
        plan = {"assign": r["assign"], "cost": r["cost"], "gib": r["gib"]}

    hist = {}
    for v in plan["assign"].values():
        hist[v] = hist.get(v, 0) + 1

    return {"available": True, "dataset": dataset, "signal": signal,
            "cand": cand, "info": info,
            "budget_gib": round(budget_gib, 4), "plan": plan,
            "hist": dict(sorted(hist.items())), "uniform_bits": uniform_bits,
            "predicted_pct": (None if signal == "alphaq"
                              else math.expm1(plan["cost"]) * 100),
            "uniform_predicted_pct": math.expm1(uni_cost) * 100,
            "bias_range": BIAS_RANGE,
            "avg_bits": round(sum(plan["assign"].values()) / len(units), 2)}


def tool_probe(warmup_s: float = 20.0) -> dict:
    """测量当前机器的显存、带宽、算力与 Roofline 平衡点。

    Args:
        warmup_s: 预热时长，单位秒。持续 matmul 直到满 ``warmup_s`` 秒。

    Returns:
        字典，含 ``name`` / ``vram_gib`` / ``bw_gbs`` / ``tflops`` / ``i_star``
        / ``throttled`` 等字段；``ok`` 为 False 时 ``reason`` 说明原因。

    Note:
        必须预热：笔记本 GPU 的功耗上限是动态的，短促测量会停在低功耗档。
        实测（RTX 5070 Laptop）不预热 20.2 TFLOPS，预热后 49.5 TFLOPS，
        即低估 2.45 倍。

    Warning:
        运行期间会占用数 GiB 显存并持续满载 ``warmup_s`` 秒以上。
    """
    import time

    import torch
    if not torch.cuda.is_available():
        return {"ok": False, "reason": "没有 CUDA"}
    prop = torch.cuda.get_device_properties(0)
    vram = prop.total_memory / GIB

    a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)

    # ---------- 预热：持续 matmul 直到满 warmup_s 秒（等功耗爬升） ----------
    # 同时采样 nvidia-smi，记录最大功耗与最大时钟
    import shutil
    import subprocess

    have_smi = shutil.which("nvidia-smi") is not None
    max_power, max_clock, last_state = 0.0, 0, ""
    t_warm = time.perf_counter()
    iters = 0
    next_sample = 0.0
    while time.perf_counter() - t_warm < warmup_s:
        a @ b
        iters += 1
        el = time.perf_counter() - t_warm
        if have_smi and el >= next_sample:
            next_sample = el + 2.0
            try:
                o = subprocess.run(
                    ["nvidia-smi", "--query-gpu=power.draw,clocks.sm,pstate,"
                     "power.max_limit", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5).stdout.strip()
                if o:
                    parts = [x.strip() for x in o.split(",")]
                    max_power = max(max_power, float(parts[0] or 0))
                    max_clock = max(max_clock, int(float(parts[1] or 0)))
                    last_state = f"pstate={parts[2]}, max_limit={parts[3]}W"
            except Exception:
                pass
    torch.cuda.synchronize()
    warm_elapsed = time.perf_counter() - t_warm

    # ---------- 算力：预热后再测 ----------
    t0 = time.perf_counter()
    for _ in range(20):
        a @ b
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / 20
    tflops = 2 * 4096 ** 3 / dt / 1e12

    # ---------- 带宽：大块拷贝（256 MB，远大于 L2） ----------
    nb = 256 * 1024 * 1024
    x = torch.empty(nb, dtype=torch.uint8, device="cuda")
    y = torch.empty_like(x)
    for _ in range(5):
        y.copy_(x)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(30):
        y.copy_(x)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / 30
    bw = 2 * nb / dt / 1e9

    i_star = tflops * 1e12 / (bw * 1e9)
    del x, y, a, b
    torch.cuda.empty_cache()
    # 功耗诊断：如果满载功耗远低于上限，说明被限制在低功耗档
    lim = 0.0
    try:
        lim = float((last_state.split("max_limit=")[-1]).rstrip("W"))
    except Exception:
        pass
    throttled = bool(lim > 0 and max_power > 0 and max_power < 0.6 * lim)
    return {"ok": True, "name": prop.name, "vram_gib": round(vram, 2),
            "bw_gbs": round(bw, 1), "tflops": round(tflops, 1),
            "i_star": round(i_star, 1), "sm": prop.multi_processor_count,
            "warmup_s": round(warm_elapsed, 1), "warmup_iters": iters,
            "max_power_w": round(max_power, 1), "max_clock_mhz": max_clock,
            "power_limit_w": lim, "throttled": throttled,
            "state": last_state}


def tool_init(model_b: float | None = None,
              model_path: str | None = None) -> dict:
    """评估当前机器与给定模型可支持的能力档位。

    Args:
        model_b: 模型参数量，单位 B（十亿参数）；``None`` 时无法估算需求。
        model_path: 本地模型路径，可为 ``None``。

    Returns:
        字典，含 ``probe``（硬件实测）/ ``need_gib_bf16``（bf16 需求，单位 GiB）
        / ``tier``（``A`` / ``B`` / ``C``）/ ``can_scan`` / ``advice``。

    Note:
        档位按 bf16 需求与显存之比划分：需求不超过显存 85% 为 ``A``，
        不超过 100% 为 ``B``，超出为 ``C``。
    """
    probe = tool_probe(warmup_s=15)
    out = {"probe": probe, "model_b": model_b, "model_path": model_path}

    if not probe.get("ok"):
        out.update(tier="C", can_scan=False,
                   advice=["没有可用 GPU，只能用参照数据。"])
        return out

    vram = probe["vram_gib"]
    # 估算 bf16 需求：参数量 × 2 bytes + 0.5 GiB 激活/上下文
    if model_b is not None:
        params = model_b * 1.029e9
    else:
        out.update(tier="?", can_scan=False,
                   advice=["需要告诉我模型规模（如 7B），或给出本地模型路径。"])
        return out

    need = params * 2 / GIB + 0.5
    out["need_gib_bf16"] = round(need, 2)
    out["vram_gib"] = vram
    out["headroom"] = round(vram / need, 2)

    if need <= vram * 0.85:
        out.update(tier="A", can_scan=True, advice=[
            f"bf16 需要 {need:.2f} GiB，你的卡 {vram:.2f} GiB，余量 "
            f"{vram/need:.2f}× —— **可以跑真实扫描**。",
            "dense 模型约 3 分钟（196 模块 × 3 位宽）；MoE 约 3 分钟（2048 专家）。",
            "跑完之后，方案里的每个数字都会是【你这台机器 + 这个模型】的实测。",
        ])
    elif need <= vram * 1.0:
        out.update(tier="B", can_scan=True, advice=[
            f"bf16 需要 {need:.2f} GiB，你的卡 {vram:.2f} GiB —— **很紧**。",
            "建议只扫一部分模块（例如每 4 个扫 1 个），其余用参照值。",
            "或者缩短评测序列到 256~512 token 以省显存。",
        ])
    else:
        out.update(tier="C", can_scan=False, advice=[
            f"bf16 需要 {need:.2f} GiB，你的卡只有 {vram:.2f} GiB —— **装不下**。",
            "**伪量化必须在 bf16 权重上做**，所以在这种情况下无法测敏感度。",
            "可选：① 用参照数据（同一模型的排序仍然有效）；"
            "② 换更小的模型；③ 用 ≥" + f"{math.ceil(need / 0.85):.0f} GiB 的卡。",
        ])
    return out


def tool_onboard(model_ref: str, bits=(4, 8), max_loss_pct: float = 1.0,
                 max_tokens: int = 801, limit: int | None = None) -> dict:
    """在给定模型上执行真实扫描，并用其自身数据生成方案。

    流程为：加载模型并检查显存是否足够 → 结构发现 → 逐单元 × 位宽扫描 →
    求解分配方案。

    Args:
        model_ref: HF 模型名（``org/name``）或本地模型路径。
        bits: 要扫描的位宽元组，默认 ``(4, 8)``。
        max_loss_pct: 方案允许的预测 PPL 损失上限，单位百分比。
        max_tokens: 评测文本截断长度，单位 token。
        limit: 限制参与扫描的单元数，``None`` 表示不限制。

    Returns:
        字典，含 ``desc`` / ``rows`` / ``plan`` / ``plans`` / ``loaded_gib``；
        ``ok`` 为 False 时 ``reason`` 说明失败原因。

    Warning:
        会真实加载模型并占用显存，并执行完整扫描（分钟量级）。
        伪量化必须在 bf16 权重上进行，bf16 装不下时无法扫描。
    """
    import torch

    from aq import onboard as ob

    if not torch.cuda.is_available():
        return {"ok": False, "reason": "没有 CUDA"}

    import transformers
    kw = ({"dtype": torch.bfloat16}
          if int(transformers.__version__.split(".")[0]) >= 5
          else {"torch_dtype": torch.bfloat16})
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_ref)
        model = AutoModelForCausalLM.from_pretrained(
            model_ref, **kw).eval().to("cuda")
    except torch.cuda.OutOfMemoryError:
        return {"ok": False, "reason":
                f"**装不下**（{model_ref} 的 bf16 超过你的显存）。\n"
                "伪量化必须在 bf16 权重上做，所以无法扫描。"
                "请换更小的模型，或用更大的卡。"}
    except Exception as e:                                  # noqa: BLE001
        return {"ok": False, "reason": f"加载失败：{type(e).__name__}: {e}"}

    used = torch.cuda.memory_allocated() / GIB
    desc = ob.describe(model)

    text = ("The history of artificial intelligence is usually dated from a summer "
            "workshop held at Dartmouth College, although the ideas that converged "
            "there had been circulating for more than a decade. Researchers believed "
            "that the central difficulty had been identified, and that the remaining "
            "work was a matter of engineering. Programs were written that could prove "
            "theorems from a small set of axioms and play checkers at a respectable "
            "level. The collapse came when the funding agencies asked for results "
            "outside those bounded worlds.")
    ids = tok(text, return_tensors="pt").input_ids[0][:max_tokens].to(model.device)

    rows = ob.scan(model, ids, bits_list=bits, describe_info=desc,
                   limit=limit, progress=False)
    try:
        plans = ob.plan_all(rows, vram_gib=None, cand=bits,
                            max_loss_pct=max_loss_pct)
    except Exception as e:                                 # noqa: BLE001
        plans = [{"name": "?", "error": str(e)}]
    plan = next((x for x in plans if "gib" in x), None) or plans[0]

    del model
    torch.cuda.empty_cache()
    return {"ok": True, "model_ref": model_ref, "desc": desc, "rows": rows,
            "plan": plan, "plans": plans, "loaded_gib": round(used, 2),
            "bits": tuple(bits), "max_loss": max_loss_pct}


def tool_speed() -> dict:
    return {"rows": [("bnb 4bit + PyTorch eager", 26.3, 2.131),
                     ("bf16 + vLLM", 203.1, 2.95),
                     ("AWQ-INT4 + vLLM", 364.8, 1.37)],
            "note": "云端 RTX 3090 实测（10.18）"}


def available_datasets() -> list[str]:
    return [k for k, v in DATASETS.items() if Path(v["path"]).exists()]


TOOLS = {"fit": tool_fit, "sensitivity": tool_sensitivity,
         "plan": tool_plan, "speed": tool_speed, "table": tool_bit_table}


# ============================================================ 能力卡片
SYSTEM_CARD = """**我是量化顾问，不是聊天机器人。**

**每个数字都带来源标记**：`实测`（真机跑过）或 `估计`（我算的）。

| 我实测过什么 | 结果 |
|---|---|
| 本机显存带宽 / 算力 | 324 GB/s / 49.5 TFLOPS → 平衡点 **153 FLOP/byte** |
| INT4 实际达到的带宽 | 只有 **9.9%**（所以 bnb 不提速）|
| **MoE 层×位宽全表** | 2bit **+3.3%~+17.4%**；3bit **−0.6%~+1.7%**（96 个测量）|
| **同预算下分配方向** | 差 **41.7%**（便宜层给 3bit vs 4bit）|
| **我的加法预测偏差** | 预测 +3.50%，实测 **+7.21%**（**低估 2.06 倍**）|
| 换推理后端 | vLLM 比 bnb eager 快 **13.9 倍** |

> ⚠️ **我会主动说清我哪里不可靠** —— 上面「低估 2.06 倍」就是。
> 所以任何方案都必须**实测验证**，不能只信预测。
"""


# ============================================================ 渲染
def _fmt(headers, rows):
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out)


def _short(name: str) -> str:
    m = re.match(r"model\.layers\.(\d+)\.(?:mlp|self_attn)\.(.*)", name)
    return f"L{m.group(1)}.{m.group(2)}" if m else name[-38:]


def _ans_fit(q: Query) -> str:
    # 已知模型（尤其是大 MoE）：直接给出装得下与否的判断
    if q.known and q.vram_gib and _MODELS:
        m = q.known
        d = _describe_model(m, q.vram_gib)
        rows = []
        for b in (16, 8, 4, 3):
            r = d["by_bits"][b]
            mark = "✅ 能" if r.get("fits") else f"❌ 差 {r['short_by']}×"
            rows.append([f"{b}bit" if b < 16 else "bf16",
                         f"{r['gib']:.1f}", mark])
        ratio = (f"\n> **MoE**：总 {m['total_b']:.0f}B / 激活 "
                 f"{m['active_b']:.0f}B = **{m['active_b'] and round(m['total_b']/m['active_b'], 1)}×**。"
                 f"**算力看激活，显存看总参数。**" if m["is_moe"] else "")
        best = next((b for b in (4, 3, 8) if d["by_bits"][b].get("fits")), None)
        verdict = (f"**能装下**，最低 {best}bit" if best
                   else f"**装不下** —— 就算 3bit 也要 "
                        f"{d['by_bits'][3]['gib']:.0f} GiB，"
                        f"你的 {q.vram_gib:.0f} GiB 差 "
                        f"{d['by_bits'][3]['gib']/q.vram_gib:.0f} 倍")
        return f"""### 结论

**{m['name']}** 在 **{q.vram_gib:.0f} GiB** 上：{verdict}

### 这个模型的关键数字（估算，非实测）

| 项 | 值 |
|---|---|
| 总参数 | **{m['total_b']:.1f} B** |
| 激活参数 | {m['active_b']:.1f} B |
| 层数 / 专家 | {m['layers']} / {m['experts'] or '—（dense）'} |
| top-k | {m['top_k'] or '—'} |
{ratio}
| 精度 | 权重 GiB | 能否装下 |
|---|---|---|
""" + "\n".join(f"| {a} | {b} | {c} |" for a, b, c in rows) + """

### 三点提醒

1. **MoE 的显存看总参数** —— 所有权重都必须驻留，激活少不代表省显存
2. **伪量化必须在 bf16 上做** —— 装不下 bf16 就没法测敏感度
3. 上表是**按参数量估算**，实测会略高（还有 KV cache 和激活）

""" + ("" if best else (
    "> ⚠️ **这个模型在你的卡上无法本地部署。** "
    "可以问「初始化」看需要多大的卡。\n"))

    if q.model_b is None or q.vram_gib is None:
        miss = []
        if q.model_b is None:
            miss.append("模型规模（如 7B）")
        if q.vram_gib is None:
            miss.append("显存（如 8G / 24G）")
        return (f"我需要知道：{'、'.join(miss)}。\n\n"
                "例如：**「我 8G 显存，想跑 7B 模型」**\n\n" + SYSTEM_CARD)
    r = tool_fit(q.model_b, q.vram_gib)
    rows = [[f"{o['bits']}bit" if o["bits"] < 16 else "bf16",
             f"{o['weight_gib']:.2f}", f"{o['need_gib']:.2f}",
             "✅ 能" if o["fits"] else "❌ 不能"] for o in r["options"]]
    ok = [o for o in r["options"] if o["fits"]]
    verdict = (f"**能装下**，最低可用 **{ok[-1]['bits']}bit**"
               if ok else "**装不下**")
    return f"""### 结论

{q.model_b:.0f}B 在 **{q.vram_gib:.0f} GiB** 上：{verdict}

### 依据（估计，非实测）

- 权重 = `0.434 GiB（lm_head） + 参数量 × 有效bits/8`
- **4bit 有效值是 4.128 bits/参数**（实测，含 scale 开销），不是 4.0
- 判断留 80% 余量

{_fmt(["精度", "权重 GiB", "含KV GiB", "能否"], rows)}

### 提醒

1. **「装得下」不等于「跑得动」** —— 还要给 KV cache 留空间
2. **bnb 的 4bit 不提速**（实测只用到 9.9% 带宽）
3. **这是估算**，具体模型要真测
"""


def _ans_sensitivity(q: Query) -> str:
    r = tool_sensitivity(q.dataset)
    if not r["available"]:
        return f"### 没有 {q.dataset} 的扫描数据（`{r['path']}`）\n"
    meta = r["meta"]
    rows = [[_short(t), f"{v:+.2f}%"] for t, v in r["top"]]
    worst = [[_short(t), f"{v:+.2f}%"] for t, v in reversed(r["worst"])]
    tail = ("> **★ 注意：单专家扰动很小（±0.75%），大部分落在「随机有益」的量级里。**\n"
            "> 所以逐专家的 α 相关性很弱（ρ=−0.15）不奇怪 —— 但**整层**尺度上信号充足。"
            if q.dataset == "moe" else
            "> **★ 单模块扰动可达 +28%（L27.down_proj），信号充足。**")
    return f"""> **粒度**：{meta['label']}（{meta['model']}，{r['n']} 个决策点）
> ⚠️ **数据来自我的机器**（RTX 5070 Laptop / 4080S），不是你的设备
> **来源：实测**（{r['bits']}bit 扫描）

### 结论

把 **{r['n']}** {meta['unit']} 各自压到 **{r['bits']}bit**：

- 最差 **{r['max']:+.2f}%**，最好 **{r['min']:+.2f}%**
- **极差 {r['spread']:.2f} 个百分点**
- **{r['n_negative']} 个是负的**（压了反而略好）

### 最敏感的 5 个

{_fmt(["模块", f"{r['bits']}bit 变化"], rows)}

### 最不敏感的 3 个

{_fmt(["模块", f"{r['bits']}bit 变化"], worst)}

### 怎么读

**负值不是错**：单点量化是一「扰动」，偶然让 PPL 略降。
**敏感度要看绝对值和跨文本一致性，不能看符号。**

{tail}
"""


def _ans_plan(q: Query) -> str:
    p = tool_plan(q.dataset, q.signal, q.cand)
    if not p.get("available"):
        return f"### 算不出方案\n\n{p.get('error', '缺少数据')}\n"

    info, hist = p["info"], p["hist"]
    hs = "  ".join(f"{b}bit×{n}" for b, n in hist.items())
    pred = p["predicted_pct"]
    uni_pred = p["uniform_predicted_pct"]
    no_ppl = pred is None
    uni_bits = p["uniform_bits"]
    lo, hi = p["bias_range"]

    # 基线：能查到实测就用实测，否则给预测
    ref = MEASURED_REF.get(q.dataset, {})
    uni_measured = ref.get(float(uni_bits))
    uni_line = (f"{uni_pred:+.2f}% | **实测 {uni_measured:+.2f}%**"
                if uni_measured is not None
                else f"{uni_pred:+.2f}% | 未实测")

    # 本方案：给区间，不给单点
    if no_ppl:
        plan_lo = plan_hi = None
    else:
        plan_lo, plan_hi = pred * lo, pred * hi
    avg = p.get("avg_bits", 0)

    sig = {"uniform": "无（均匀基线）", "alphaq": "α（免校准谱指标）",
           "measured": "实测 ΔPPL", "hybrid": "混合"}.get(q.signal, q.signal)
    note = ("\n\n> " + "\n> ".join(q.notes)) if q.notes else ""
    prov = ("\n> **数据来源**：敏感度是我在 RTX 5070 Laptop / 4080S 上测的参照值；\n"
            "> 你的带宽/算力会不同 —— 问「**测一下我的设备**」可以现测。\n")

    return f"""{prov}> **粒度**：{info['label']}（{info['n']} 个单元）
> **信号**：{sig}　**候选位宽**：{sorted(info['bits'])}
> **预算**：{p['budget_gib']:.4f} GiB（平均 **{avg}** bit）

### 方案

**位宽分布**：`{hs}`

### ★ 预测 vs 实测（这段最重要）

| | 预测 | 实际 |
|---|---|---|
| 均匀 {uni_bits}bit（基线）| {uni_line} |
| **本方案（avg {avg} bit）** | {_fmt_pred(pred, plan_lo, plan_hi)} |

> **为什么不给单点**：我实测过自己的加法模型会**系统性低估**，而且
> **低估倍数取决于方案被"挑"的程度**：
> - 均匀 4bit：预测 +4.30% → 实测 +4.73%（**1.10×**）
> - DP 择优方案：预测 +3.50% → 实测 +7.21%（**2.06×**）
>
> **⇒ 优化器越"挑"，低估越严重。所以预测只能用来排序。**

### 三条提醒

1. **2bit 是悬崖**（实测每层掉 3.3%~17.4%）—— 任何方案都不该用它
2. **3bit 几乎无损**（−0.6%~+1.7%），真正的决策区间是 {{3,4,5}}
3. **同预算下分配方向有 41.7% 的影响**，但必须建立在实测排序上

### 我做不到的

- MoE 只有 **2bit 的逐专家**数据，所以方案是**逐层**粒度
- 没有端到端部署验证（伪量化不落地）
{note}
"""


def _fmt_pred(pred, lo, hi) -> str:
    if pred is None:
        return "**无法换算成 ΔPPL**（α 信号的代价是噪声方差单位，不是 PPL）"
    return f"**{pred:+.2f}%** | 预计 **{lo:+.2f}% ~ {hi:+.2f}%**"


def _ans_onboard(q: Query) -> str:
    import statistics
    if not q.model_ref:
        return ("### 需要模型名称或路径\n\n"
                "例如：**「量化我的模型 Qwen/Qwen2.5-1.5B-Instruct」**\n"
                "或 **「量化我的模型 D:\\models\\my-3b」**\n\n"
                "⚠️ 会真的加载模型并扫描，**需要模型装得进 bf16**（约 2 分钟）。")
    bits = tuple(int(x) for x in ("2", "4", "8") if True)  # 实际取 (4, 8)，见下一行
    bits = (4, 8)
    r = tool_onboard(q.model_ref, bits=bits, max_loss_pct=1.0)
    if not r.get("ok"):
        return f"### 扫描失败\n\n{r.get('reason')}\n"

    d, rows, plan = r["desc"], r["rows"], r["plan"]
    agg = {}
    for x in rows:
        agg.setdefault(x["bits"], []).append(x["rel_pct"])
    tbl = "\n".join(
        f"| {b}bit | {min(v):+.2f}% | {statistics.median(v):+.2f}% | "
        f"{max(v):+.2f}% | {len(v)} |"
        for b, v in sorted(agg.items()))
    top = sorted((x for x in rows if x["bits"] == min(bits)),
                 key=lambda x: -x["rel_pct"])[:6]
    topline = "\n".join(
        f"| `{x['target'][-44:]}` | {x['rel_pct']:+.2f}% |" for x in top)

    plans = r.get("plans", [])
    if plans:
        md = ["| 方案 | 场景 | 平均位宽 | 占用 | 预测 ΔPPL |",
              "|---|---|---|---|---|"]
        for x in plans:
            if "error" in x:
                md.append(f"| {x['name']} | {x['scene']} | — | 求解失败 | {x['error']} |")
            else:
                md.append(f"| **{x['name']}** | {x['scene']} | "
                          f"{x.get('avg_bits')} bit | **{x['gib']:.2f} GiB** | "
                          f"{x['predicted_pct']:+.2f}% |"
                          if x.get("predicted_pct") is not None
                          else f"| **{x['name']}** | {x['scene']} | "
                               f"{x.get('avg_bits')} bit | **{x['gib']:.2f} GiB** | — |")
        plan_table = "\n".join(md)
    else:
        plan_table = "无可用方案"
    plan_md = plan_table
    if "hist" in plan:
        plan_md = (f"- 目标：**{plan.get('mode')}**（预测损失 ≤ "
                   f"{r['max_loss']}% 下压到最低位宽）\n"
                   f"- 实际占用：**{plan['gib']:.3f} GiB**"
                   f"（平均 **{plan.get('avg_bits')}** bit）\n"
                   f"- 位宽分布：`{plan['hist']}`\n"
                   f"- **预测 ΔPPL：{plan['predicted_pct']:+.2f}%**")

    return f"""### ✅ 在 `{q.model_ref}` 上真跑完了

> **这是你自己这个模型的实测**，不是参照值。
> 加载占用 {r['loaded_gib']:.2f} GiB ｜ 可量化单元 **{d['n_units']}** ｜ 扫描 {len(rows)} 条

**① 结构**

| 项 | 值 |
|---|---|
| 参数量 | {d['params_total']:,} |
| 可量化参数 | {d['params_quant']:,} |
| 普通 Linear | **{d['n_linear']}** 个 |
| 融合专家张量 | **{d['n_expert']}** 个 |

**② 实测敏感度**

| 位宽 | 最小 | 中位 | 最大 | 单元数 |
|---|---|---|---|---|
{tbl}

最敏感的 6 个（{min(bits)}bit）：

| 单元 | ΔPPL |
|---|---|
{topline}

**③ 可选方案（挑一个，我按它量化导出）**

{plan_md}

> ⚠️ 预测误差约 1~2 倍、**方向还不一定**（K005/K006）——
> 最终方案仍需在完整评估集上实测。

"""


def _ans_init(q: Query) -> str:
    r = tool_init(q.model_b)
    pr = r.get("probe", {})
    if not pr.get("ok"):
        return f"### 初始化失败\n\n{pr.get('reason')}\n"
    tier = r.get("tier", "?")
    badge = {"A": "✅ **A 档：可以跑真实扫描**",
             "B": "⚠️ **B 档：显存很紧，建议部分扫描**",
             "C": "❌ **C 档：装不下 bf16，只能用参照数据**",
             "?": "❓ 需要模型规模"}.get(tier, tier)
    need = r.get("need_gib_bf16")
    rows = ""
    if need is not None:
        rows = (f"\n| bf16 需求 | **{need:.2f} GiB** |\n"
                f"| 你的显存 | {r['vram_gib']:.2f} GiB |\n"
                f"| 余量 | **{r['headroom']:.2f}×** |\n")
    advice = "\n".join(f"{i+1}. {a}" for i, a in enumerate(r.get("advice", [])))
    return f"""### 初始化结果：{badge}

**① 你的硬件（现测）**

| 项 | 值 |
|---|---|
| GPU | **{pr['name']}** |
| 显存 | {pr['vram_gib']:.2f} GiB |
| 带宽 | **{pr['bw_gbs']:.1f} GB/s** |
| 算力 (bf16) | **{pr['tflops']:.1f} TFLOPS** |
| 平衡点 I∗ | **{pr['i_star']:.1f} FLOP/byte** |
| 满载功耗 | {pr['max_power_w']:.1f} / {pr['power_limit_w']:.0f} W |
{rows}
**② 结论**

{advice}

### ③ 为什么"敏感度"可以复用、而"显存/算力"不行

| 量 | 由什么决定 | 能否跨机器复用 |
|---|---|---|
| 显存 / 带宽 / 算力 / 功耗 | **你的硬件** | ❌ 必须现测 |
| **敏感度排序（哪些层能压）** | **模型权重数值** | ✅ **同一模型可复用** |

> **★ 也就是说：换机器不用重测敏感度，但换模型必须重测。**

"""


def _ans_probe(q: Query) -> str:
    p_ = tool_probe()
    if not p_.get("ok"):
        return f"### 探测失败\n\n{p_.get('reason')}\n"
    # 多少 token/s 的天花板（decode 带宽受限）
    ceil = p_["bw_gbs"] * 1e9 / (2 * 1e9 * 8 / 8) / 1  # 占位，下面重算
    return f"""### 这台机器的实测参数

> **来源：刚在你的设备上现测的**（不是读我的 CSV）
> **已预热 {p_['warmup_s']:.0f} 秒**（{p_['warmup_iters']} 次 matmul）—— 笔记本 GPU 不预热会低估算力 2 倍以上

| 项 | 值 |
|---|---|
| GPU | **{p_['name']}** |
| 显存 | **{p_['vram_gib']:.2f} GiB** |
| SM 数 | {p_['sm']} |
| 实测带宽 | **{p_['bw_gbs']:.1f} GB/s** |
| 实测算力（bf16） | **{p_['tflops']:.1f} TFLOPS** |
| **Roofline 平衡点 I∗** | **{p_['i_star']:.1f} FLOP/byte** |

### 功耗诊断

| 项 | 值 |
|---|---|
| 预热期间最大功耗 | **{p_['max_power_w']} W** / 上限 {p_['power_limit_w']} W |
| 预热期间最大 SM 时钟 | **{p_['max_clock_mhz']} MHz** |
| 报错状态 | {p_['state'] or '未采到'} |

{('> ⚠️ **这台机器被限制在低功耗档**（满载功耗不到上限的 60%）。\n'
  '> 真实算力可能是上面的 **2 倍以上**。\n'
  '> **→ 请插上电源、把电源计划切成"高性能"，然后重测。**\n'
  '> （我自己在同一台机器上测过：低功耗档 20.4 TFLOPS，满功耗档 49.5 TFLOPS，差 2.4 倍。）')
 if p_['throttled'] else '> ✅ 功耗已接近上限，这组数可信。'}

### 这组数怎么用

- **I∗ = {p_['i_star']:.0f}** 意味着：算术强度低于这个值的算子（decode 大约只有 1~4）
  **是带宽受限的** —— 所以位宽降得再低也提不了速，**换后端才有效**
- **能装多大的模型**：`显存 × 80% ÷ 2 bytes` ≈
  **{p_['vram_gib'] * 0.8 / 2:.1f} B 参数（bf16）**，
  **{p_['vram_gib'] * 0.8 / 0.5:.1f} B 参数（4bit）**

### ⚠️ 一句必须说清的话

**上面是"你的机器"的实测；而敏感度数据（哪些层能压）是我在
RTX 5070 Laptop 和 4080S 上测的"参照值"。**

- 带宽 / 算力 / 平衡点：**与你的设备有关，必须现测** ✅ 上面已经测了
- 敏感度排序：**与设备基本无关**（它取决于权重数值，不取决于硬件），
  但严格来说要在你的模型上测

"""


def _ans_knowledge(q: Query) -> str:
    if not _KB:
        return "### 知识库缺失\n\n需要 `knowledge/claims.json` 和 `aq/knowledge.py`。\n"
    hits = _kb_query(q.raw, top=5)
    st = _kb_stats()
    head = (f"> **知识库**：{st['total']} 条事实，其中 "
            f"**{st['by_confidence'].get('high', 0)} 条是我自己实测的**。\n"
            f"> 每条带条件、证据、置信度；矛盾显式并列，不隐藏。\n")
    if not hits:
        return head + "\n没找到相关事实。试试问「α 指标可不可靠」或「2bit 能不能用」。\n"
    body = cite_block = ""
    if _KB:
        cite_block = _cite([c["id"] for c in hits])
    return head + "\n### 相关事实\n\n" + cite_block + "\n" + _conflicts_block()


def _cite_for(intent: str, dataset: str) -> str:
    """按意图给回答末尾挂引用"""
    if not _KB:
        return ""
    ids = []
    if intent == "sensitivity":
        ids = ["K002", "K007"] if dataset == "moe" else ["K020", "K001"]
    elif intent == "plan":
        ids = ["K004", "K005", "K002"] if dataset == "moe" else ["K018", "K001"]
    elif intent == "table":
        ids = ["K002", "K003", "K010", "K019"]
    elif intent == "speed":
        ids = ["K013", "K014", "K015"]
    return ("\n" + _cite(ids, show_evidence=False)) if ids else ""


def _ans_table(q: Query) -> str:
    tt = tool_bit_table()
    if not tt.get("available"):
        return f"### 没有层×位宽表（`{BIT_TABLE}`）\n"
    bits = tt["bits"]
    rows = [[L] + [f"{tt['table'][L].get(b, float('nan')):+.2f}%"
                   for b in bits] for L in sorted(tt["table"])]
    ext = []
    for b in bits:
        vals = [(tt["table"][L][b], L) for L in tt["table"]
                if b in tt["table"][L]]
        ext.append((min(vals), max(vals)))
    summary = "\n".join(
        f"| {b} | L{ext[i][0][1]} {ext[i][0][0]:+.2f}% | "
        f"L{ext[i][1][1]} {ext[i][1][0]:+.2f}% |"
        for i, b in enumerate(bits))
    return f"""> **来源：实测**（OLMoE-1B-7B，16 层 × {len(bits)} 位宽 = {len(bits)*16} 个测量）

### 结论：**2bit 是悬崖，3bit 几乎无损**

| 位宽 | 最好层 | 最差层 |
|---|---|---|
{summary}

### 完整表格

{_fmt(["层"] + bits, rows)}

### 怎么读

- **2bit 整列都在 +3%~+17%**，没有一层能幸免 → **禁用 2bit**
- **3bit 整列都在 ±1.8% 内** → **3bit 是安全下限**
- 层趋势只在 2bit 明显（ρ=+0.76），3bit 消失（ρ=+0.16）
"""


def _ans_speed(q: Query) -> str:
    r = tool_speed()
    base = r["rows"][0][1]
    rows = [[n, f"{t:.1f}", f"{m:.2f}"] for n, t, m in r["rows"]]
    return f"""### 结论：换后端比换精度重要得多

**{r['note']}**

{_fmt(["配置", "吞吐 tok/s", "权重 GiB"], rows)}

- vLLM vs bnb eager：**{r['rows'][1][1]/base:.1f} 倍**（只换后端）
- INT4 vs bf16（同在 vLLM）：**{r['rows'][2][1]/r['rows'][1][1]:.2f} 倍**

### 为什么

decode 是带宽受限的（算术强度 ≈ 1，本机平衡点 153 FLOP/byte）。
**bnb 的 4bit 只用 9.9% 带宽** —— 时间花在反量化和 kernel 下发上。

### 一个必须说清的限制

vLLM 的量化是**全局配置**，**不支持逐专家位宽**。
所以逐层/逐专家方案不能直接在上面跑，只能取整成均匀 4bit。
"""


# ============================================================ 入口
def llm_selftest() -> str:
    """LLM 解析层的自检：配置、连通性、一次真实解析"""
    if not _LLM:
        return "❌ `aq/llm_parse.py` 导入失败"
    if not _llm_available():
        return ("⚠️ **未配置 API key** —— 现在走规则路由（也能跑，但不够聪明）。\n\n"
                "**① DeepSeek 官方**（最便宜）\n"
                "```powershell\n"
                "$env:LLM_API_KEY = \"sk-xxxx\"\n"
                "$env:LLM_BASE_URL = \"https://api.deepseek.com\"\n"
                "$env:LLM_MODEL = \"deepseek-chat\"\n"
                "```\n"
                "**② 硅基流动**（国内备用）\n"
                "```powershell\n"
                "$env:LLM_API_KEY = \"sk-xxxx\"\n"
                "$env:LLM_BASE_URL = \"https://api.siliconflow.cn/v1\"\n"
                "$env:LLM_MODEL = \"Qwen/Qwen2.5-7B-Instruct\"\n"
                "```\n"
                "**③ ★ 本地 Ollama**（完全离线，现场最稳）\n"
                "```powershell\n"
                "ollama run qwen2.5:3b\n"
                "$env:LLM_API_KEY = \"ollama\"\n"
                "$env:LLM_BASE_URL = \"http://localhost:11434/v1\"\n"
                "$env:LLM_MODEL = \"qwen2.5:3b\"\n"
                "```\n"
                "配好后**重开终端和 Gradio**。没配也不影响其他功能（走规则）。")
    import time
    t0 = time.perf_counter()
    d = _llm_parse("我16G显存，想本地部署deepseekV3模型")
    dt = time.perf_counter() - t0
    if d is None:
        return (f"❌ **key 已配置，但调用失败**（{dt:.1f}s）。\n\n"
                "检查：网络能否访问 api.deepseek.com、key 是否有效、余额是否足够。")
    return (f"✅ **LLM 解析正常**（{dt:.2f}s）\n\n"
            f"输入：我16G显存，想本地部署deepseekV3模型\n\n"
            f"解析结果：`{d}`\n\n"
            "> 注意：LLM 只解析意图和参数，**所有数值结论仍来自实测**。",
            )


def answer(text: str, context: str | None = None) -> str:
    q = parse_query(text, context=context)
    if q.intent == "selftest":
        return llm_selftest()
    if q.intent == "onboard":
        return _ans_onboard(q)
    if q.intent == "init":
        return _ans_init(q)
    if q.intent == "probe":
        return _ans_probe(q)
    if q.intent == "knowledge":
        return _ans_knowledge(q)
    if q.intent == "sensitivity":
        return _ans_sensitivity(q) + _cite_for("sensitivity", q.dataset)
    if q.intent == "speed":
        return _ans_speed(q) + _cite_for("speed", q.dataset)
    if q.intent == "table":
        return _ans_table(q) + _cite_for("table", q.dataset)
    if q.intent == "plan":
        return _ans_plan(q) + _cite_for("plan", q.dataset)
    return _ans_fit(q)

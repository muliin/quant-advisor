"""统一命令行入口。

各子命令均可独立运行，便于单独调试；``run`` 只负责把它们串联起来。

用法
----
    python adapt_quant.py cost --matrix
    python adapt_quant.py cost --model 1.5B --bits 4 --seq 2048
    python adapt_quant.py bench --model Qwen/Qwen2.5-1.5B-Instruct
    python adapt_quant.py analyze --model ... --out sens.csv
    python adapt_quant.py solve --sens sens.csv --budget 1.2
    python adapt_quant.py pareto --sens sens.csv --out pareto.png
    python adapt_quant.py run --model ... --budget auto --out runs/qwen1.5b/

所有参数均可由命令行覆盖，切换实验环境无需修改代码。未实现的子命令
返回退出码 3 并给出提示，不抛出 traceback。
"""

from __future__ import annotations

import argparse
import sys


# ============================================================================
# 子命令实现
# ============================================================================
def cmd_cost(args) -> int:
    from aq import cost

    if args.matrix:
        print(cost.print_fit_matrix(headroom=args.headroom))
        return 0

    # 单模型
    n_params = args.n_params or cost.MODEL_SIZES.get(args.model_size)
    if n_params is None:
        print(f"未知模型规模 {args.model_size!r}；可选：{list(cost.MODEL_SIZES)} "
              f"或用 --n-params 直接给参数量", file=sys.stderr)
        return 2

    d = cost.total_gib(
        n_params=n_params, bits=args.bits,
        num_layers=args.layers, n_kv_heads=args.n_kv,
        head_dim=args.head_dim, seq_len=args.seq, batch=args.batch,
    )
    print(f"模型      : {args.model_size}  ({n_params:,} 参数)")
    print(f"精度      : {args.bits} bit")
    print(f"序列长度  : {args.seq}   batch={args.batch}")
    print("-" * 46)
    print(f"  权重     : {d['weight_gib']:8.4f} GiB")
    print(f"  KV cache : {d['kv_cache_gib']:8.4f} GiB")
    print(f"  余量     : {d['overhead_gib']:8.4f} GiB")
    print(f"  ------------------------------")
    print(f"  合计     : {d['total_gib']:8.4f} GiB")

    g = cost.min_gpu(d["total_gib"], headroom=args.headroom)
    print(f"\n最低卡型  : {g.name} （按 {args.headroom:.0%} 余量）")
    print(f"\n注意：以上只计算显存占用。速度提升需在带宽充足 + 低开销后端"
          f"（4090 + vLLM）验证，本机不可得。")
    return 0


def cmd_bench(args) -> int:
    from aq import bench
    try:
        r = bench.bench_model(args.model, args.prompt_len, args.max_new_tokens)
    except NotImplementedError as e:
        print(f"未实现：{e}", file=sys.stderr)
        print("   性能基准子命令尚未接入", file=sys.stderr)
        return 3
    for k, v in r.items():
        print(f"  {k:16}: {v}")
    return 0


def cmd_analyze(args) -> int:
    from aq import sensitivity
    try:
        rows = sensitivity.scan(args.model, bits_list=tuple(args.bits))
    except NotImplementedError as e:
        print(f"未实现：{e}\n   敏感度扫描子命令尚未接入", file=sys.stderr)
        return 3
    sensitivity.to_csv(rows, args.out)
    print(f"已写出 {args.out}")
    return 0


def cmd_solve(args) -> int:
    from aq import sensitivity, allocate
    try:
        sens = sensitivity.from_csv(args.sens)
        plan = allocate.solve(sens, budget_gib=args.budget)
    except NotImplementedError as e:
        print(f"未实现：{e}\n   该子命令尚未接入", file=sys.stderr)
        return 3
    import json
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(plan, f, ensure_ascii=False, indent=2)
    print(f"已写出 {args.out}（平均位宽 {plan['avg_bits']:.2f} bit）")
    return 0


def cmd_pareto(args) -> int:
    from aq import sensitivity, pareto
    try:
        sens = sensitivity.from_csv(args.sens)
        rows = pareto.sweep_budget(sens, n_points=args.points)
        pareto.plot(rows, out_path=args.out)
    except NotImplementedError as e:
        print(f"未实现：{e}\n   该子命令尚未接入", file=sys.stderr)
        return 3
    print(f"已写出 {args.out}")
    return 0


def cmd_run(args) -> int:
    """串联 analyze、solve、pareto、cost 四个子命令；当前仅打印执行计划。"""
    print("=" * 60)
    print(f"自适应量化：{args.model}")
    print("=" * 60)
    steps = ["analyze", "solve", "pareto", "cost"]
    print(f"将执行 {len(steps)} 步：{' → '.join(steps)}")
    print(f"输出目录：{args.out}")
    print("\n注意：当前各子命令需单独调用，端到端串联尚未接入。")
    print("   现在请分步执行各子命令，例如：")
    print(f"     python adapt_quant.py analyze --model {args.model} --out {args.out}/sens.csv")
    return 0


# ============================================================================
# 参数解析
# ============================================================================
def build_parser() -> argparse.ArgumentParser:
    from aq import cost

    ap = argparse.ArgumentParser(
        prog="adapt_quant",
        description="自适应混合精度量化工具",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    # ---- cost ----
    p = sub.add_parser("cost", help="显存 → 卡型 → 年成本")
    p.add_argument("--matrix", action="store_true", help="打印模型规模 × 精度 × 卡型矩阵")
    p.add_argument("--model-size", default="1.5B", choices=list(cost.MODEL_SIZES))
    p.add_argument("--n-params", type=int, default=None, help="直接指定参数量（覆盖 --model-size）")
    p.add_argument("--bits", type=float, default=16, help="平均位宽，可以是 4/8/16 或混合方案的平均值")
    p.add_argument("--seq", type=int, default=2048, help="序列长度（影响 KV cache）")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--layers", type=int, default=28)
    p.add_argument("--n-kv", type=int, default=2, help="KV 头数（GQA）")
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--headroom", type=float, default=cost.DEFAULT_HEADROOM,
                   help="模型最多占卡显存的比例（默认 0.8）")
    p.set_defaults(func=cmd_cost)

    # ---- bench ----
    p = sub.add_parser("bench", help="速度测量")
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--prompt-len", type=int, default=128)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.set_defaults(func=cmd_bench)

    # ---- analyze ----
    p = sub.add_parser("analyze", help="逐层敏感度扫描")
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--bits", type=int, nargs="+", default=[8, 4])
    p.add_argument("--out", default="sensitivity.csv")
    p.set_defaults(func=cmd_analyze)

    # ---- solve ----
    p = sub.add_parser("solve", help="给定预算求位宽分配")
    p.add_argument("--sens", default="sensitivity.csv")
    p.add_argument("--budget", type=float, required=True, help="显存预算（GiB，含权重）")
    p.add_argument("--out", default="plan.json")
    p.set_defaults(func=cmd_solve)

    # ---- pareto ----
    p = sub.add_parser("pareto", help="生成 Pareto 前沿")
    p.add_argument("--sens", default="sensitivity.csv")
    p.add_argument("--points", type=int, default=12)
    p.add_argument("--out", default="pareto.png")
    p.set_defaults(func=cmd_pareto)

    # ---- run ----
    p = sub.add_parser("run", help="一条龙")
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--budget", default="auto")
    p.add_argument("--out", default="runs/latest")
    p.set_defaults(func=cmd_run)

    return ap


def main(argv: list[str] | None = None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

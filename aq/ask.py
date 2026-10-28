"""命令行问答入口。

把自然语言查询交给 :func:`aq.agent.answer` 渲染回答；
``--json`` 只输出结构化解析字段，不渲染 markdown，便于其他程序消费。

用法：

    python -m aq.ask "8G 显存能跑 Qwen/Qwen2.5-1.5B-Instruct 吗"
    python -m aq.ask "MoE 的哪些专家不能压"
    python -m aq.ask --json "给出一个 MoE 分配方案"

返回码：缺少查询参数时打印本帮助文本并返回 2，其余路径返回 0。
"""
from __future__ import annotations

import json
import sys

from aq.agent import SYSTEM_CARD, answer, parse_query


def main(argv: list[str]) -> int:
    args = [a for a in argv if not a.startswith("--")]
    flags = {a for a in argv if a.startswith("--")}
    if not args:
        print(__doc__)
        return 2

    text = " ".join(args)
    q = parse_query(text)

    if "--json" in flags:
        # 只输出解析结果与关键字段，不渲染 markdown（便于其他 agent 消费）
        print(json.dumps({
            "intent": q.intent,
            "model_b": q.model_b,
            "vram_gib": q.vram_gib,
            "dataset": q.dataset,
            "signal": q.signal,
            "cand": q.cand,
            "known_model": q.known["name"] if q.known else None,
            "model_ref": q.model_ref,
            "notes": q.notes,
        }, ensure_ascii=False, indent=2))
        return 0

    if "--card" in flags:
        print(SYSTEM_CARD)
        return 0

    print(answer(text))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

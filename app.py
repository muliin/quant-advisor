"""app.py — 量化顾问 Agent 的 Gradio 界面

运行
    python app.py                 # 默认 http://127.0.0.1:7860
    python app.py --share         # 生成公网链接（比赛"作品体验链接"用这个）

设计（见 10.27 教材）
  · 回答全部由 aq.agent 生成，**不依赖任何 LLM API**（现场必须能跑）
  · 每个回答都内联"我实测了什么"——这是区别于通用 Agent 的核心
  · 右侧固定展示项目的硬数据（评委一眼能看到）
"""
from __future__ import annotations

import argparse

import gradio as gr

from aq.agent import SYSTEM_CARD, answer, tool_probe

EXAMPLES = [
    "我 8G 显存，想跑 7B 模型",
    "MoE 的哪些专家不能压",
    "给我一个 MoE 的分配方案",
    "MoE 每层压到不同位宽掉多少",
    "用 vLLM 能快多少",
    "α 指标可不可靠",
    "你这个结论有什么依据",
    "测一下我的设备",
    "量化我的模型 Qwen/Qwen2.5-1.5B-Instruct",
]

SIDEBAR = """### 这个 Agent 和其他有什么不同

**通用 Agent 会说**："用 4bit 加载即可，大概省 75% 显存。"

**这个 Agent 会说**："你的第 15 层压到 2bit 会掉 17.4%，
第 1 层只掉 3.3% —— 所以先压前面。"

差别在括号里的数字 —— **它们都是我在真机上测的**。

---

### 实测清单（20 条事实，18 条自测）

| 项目 | 结果 |
|---|---|
| 本机带宽 / 算力 | 324 GB/s / 49.5 TFLOPS |
| Roofline 平衡点 | **153 FLOP/byte** |
| INT4 实际带宽利用 | **9.9%**（所以 bnb 不提速）|
| **MoE 层×位宽（96 个测量）** | 2bit **+3.3%~+17.4%**；3bit **−0.6%~+1.7%** |
| **同预算下分配方向** | 差 **41.7%** |
| **我的加法预测偏差** | 预测 +3.50%，实测 **+7.21%**（**低估 2.06×**）|
| vLLM vs bnb eager | **13.9 倍** |

---

### ⚠️ 已知局限（我要主动说）

- 我的**加法模型会系统性低估**（2.06 倍）—— 任何方案都要实测验证
- MoE 只有 **2bit 的逐专家**数据，所以方案是逐层粒度
- vLLM 不支持逐专家位宽，细粒度方案只能粗粒度落地
- 单模块敏感度**不能简单相加**（我验证过，贪心因此失效）

---

### 📚 可以问「依据」

问「**你这个结论有什么依据**」或「**α 指标可不可靠**」，
我会列出**带条件、带证据、带置信度**的事实条目，
以及**我记录到的矛盾**（不隐藏）。
"""


def _last_user(history) -> str | None:
    """取上一轮用户说的话，作为上下文"""
    if not history:
        return None
    for turn in reversed(history):
        try:
            role, content = turn[0], turn[1]
        except Exception:                                  # noqa: BLE001
            continue
        if role in ("user", "用户") and content:
            return content
    return None


def respond(message: str, history: list):
    """收到一句话 → 返回 (清空输入, 新历史, 证据面板)"""
    if not message or not message.strip():
        return "", list(history), SYSTEM_CARD
    reply = answer(message, context=_last_user(history))
    msgs = list(history) + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": reply},
    ]
    return "", msgs, SYSTEM_CARD


def device_card() -> str:
    """启动时探测本机，失败就给出提示"""
    try:
        p = tool_probe(warmup_s=15)
    except Exception as e:                                  # noqa: BLE001
        return f"### 设备探测失败\n\n`{type(e).__name__}: {e}`"
    if not p.get("ok"):
        return f"### 设备探测失败\n\n{p.get('reason')}"
    warn = ("\n> ⚠️ **功耗只有 %.0f / %.0f W —— 这台机器被限功耗了，"
            "真实算力可能是上面的 2 倍以上。请插电并切到「高性能」。**"
            % (p["max_power_w"], p["power_limit_w"])) if p["throttled"] else ""
    return f"""### 你的设备（启动时现测）

| 项 | 值 |
|---|---|
| GPU | **{p['name']}** |
| 显存 | {p['vram_gib']:.2f} GiB |
| 带宽 | **{p['bw_gbs']:.1f} GB/s** |
| 算力 (bf16) | **{p['tflops']:.1f} TFLOPS** |
| 平衡点 I∗ | **{p['i_star']:.1f} FLOP/byte** |
| 满载功耗 | {p['max_power_w']:.1f} / {p['power_limit_w']:.0f} W |
{warn}"""


def build() -> gr.Blocks:
    with gr.Blocks(title="量化顾问 Agent") as demo:
        gr.Markdown(
            "# 量化顾问 Agent\n"
            "**给定模型和显存，告诉你哪些层能压、能省多少、该不该换后端。**\n\n"
            "所有数字都来自实测，不是训练数据里的说法。"
        )
        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(label="对话", height=480)
                with gr.Row():
                    msg = gr.Textbox(
                        label="你的问题", scale=4,
                        placeholder="例如：我 8G 显存，想跑 7B 模型")
                    send = gr.Button("发送", variant="primary", scale=1)
                gr.Examples(examples=EXAMPLES, inputs=msg, label="试试这些")
                clear = gr.Button("清空")
            with gr.Column(scale=2):
                side = gr.Markdown(SIDEBAR)
                dev = gr.Markdown(device_card())
                evid = gr.Markdown(SYSTEM_CARD)

        send.click(respond, [msg, chatbot], [msg, chatbot, evid])
        msg.submit(respond, [msg, chatbot], [msg, chatbot, evid])
        clear.click(lambda: ([], SYSTEM_CARD), None, [chatbot, evid])
    return demo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--share", action="store_true", help="生成公网链接")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()
    build().launch(share=args.share, server_port=args.port)


if __name__ == "__main__":
    main()

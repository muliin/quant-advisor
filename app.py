"""app.py — 量化顾问 Agent 的 Gradio 界面

运行
    python app.py                 # 默认 http://127.0.0.1:7860
    python app.py --share         # 生成公网临时链接（72 小时有效）

布局原则
  首屏只放功能：标题 → 对话 → 提问框 → 示例。
  参考数据、设备探测、能力说明一律收进折叠区，放在下方。
  API Key 折叠区紧跟标题，一眼可见，但不占首屏空间。

依赖
  回答由 ``aq.agent`` 生成。未配置 API key 时走规则路由与模板回答，
  配置后由 LLM 组织措辞，输出仍须通过数字校验。
"""
from __future__ import annotations

import argparse

# INTERPRETER HINT: 若在此报 ModuleNotFoundError，说明用的不是项目环境。
try:
    import gradio as gr
except ModuleNotFoundError:
    import sys
    sys.stderr.write(
        "\n[启动失败] 当前解释器缺少 gradio：\n"
        f"  {sys.executable}\n\n"
        "请使用项目环境的解释器启动，例如：\n"
        "  D:\\miniforge3\\envs\\llmquant\\python.exe app.py\n"
        "或先激活环境：conda activate llmquant && python app.py\n\n"
        "依赖安装：pip install -r requirements.txt\n")
    raise SystemExit(1)

from aq.agent import SYSTEM_CARD, answer, tool_probe
from aq.llm_parse import available as llm_available

CSS = """
.gradio-container { max-width: 980px !important; margin: 0 auto !important; }
footer { display: none !important; }
#hdr { text-align: center; margin-bottom: 4px; }
#hdr p { margin: 4px 0; }
"""

# 首屏示例：只放最能一眼看懂功能的四条
EXAMPLES = [
    "我 8G 显存，想跑 Qwen/Qwen2.5-1.5B-Instruct",
    "MoE 的哪些专家不能压",
    "给我一个 MoE 的分配方案",
    "MoE 每层压到不同位宽掉多少",
    "用 vLLM 能快多少",
    "你这个结论有什么依据",
]

DATA_MD = """### 实测清单

| 项目 | 结果 |
|---|---|
| 本机带宽 / 算力 | 324 GB/s / 49.5 TFLOPS |
| Roofline 平衡点 | **153 FLOP/byte** |
| INT4 实际带宽利用 | **9.9%**（所以 bnb 不提速）|
| MoE 层×位宽（96 个测量）| 2bit **+3.3%~+17.4%**；3bit **−0.6%~+1.8%** |
| 同预算下分配方向 | 差 **41.7%** |
| 加法代价模型偏差 | 预测 +3.50%，实测 **+7.21%**（低估 2.06×）|
| vLLM vs bnb eager | **13.9 倍** |

### 已知局限

- 加法代价模型会**系统性低估**（约 2 倍）—— 任何方案都要实测验证
- MoE 只有 **2bit 的逐专家**数据，所以方案是逐层粒度
- vLLM 不支持逐专家位宽，细粒度方案只能粗粒度落地
- 单模块敏感度**不能简单相加**（贪心因此失效）
- 伪量化必须在 bf16 权重上做，模型装不下 bf16 时无法测敏感度

### 可以问「依据」

问「**你这个结论有什么依据**」或「**α 指标可不可靠**」，
会列出**带条件、带证据、带置信度**的事实条目，以及**记录到的矛盾**。
"""

WELCOME = """## 量化顾问

给定模型和显存预算，**实测**每一层能压到几位、会掉多少，并给出可执行的方案。

在下面输入框提问，或点一个示例开始。
"""


def _last_user(history) -> str | None:
    """取上一轮用户说的话，作为上下文。"""
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
    """处理一次提问。

    Returns:
        ``(清空后的输入框, 新的对话历史)``。
    """
    if not message or not message.strip():
        return "", list(history)
    reply = answer(message, context=_last_user(history))
    msgs = list(history) + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": reply},
    ]
    return "", msgs


def respond_with_key(query: str, history, api_key: str = ""):
    """访客填了 key 就临时覆盖环境变量，用完立即恢复。"""
    import os
    prev = os.environ.get("LLM_API_KEY")
    if api_key and api_key.strip():
        os.environ["LLM_API_KEY"] = api_key.strip()
    try:
        return respond(query, history)
    finally:
        if prev is None:
            os.environ.pop("LLM_API_KEY", None)
        else:
            os.environ["LLM_API_KEY"] = prev


def llm_status(api_key: str = "") -> str:
    """显示当前表达能力来源。"""
    if api_key and api_key.strip():
        return "🟢 使用你填入的 Key"
    if llm_available():
        return "🟢 已启用（服务端配置）"
    return "⚪ 规则模式 · 未使用外部 API"


def device_panel(api_key: str = "") -> str:                 # noqa: ARG001
    """点击按钮时才探测设备（探测需约 20 秒，不放在启动路径上）。"""
    try:
        p = tool_probe(warmup_s=15)
    except Exception as e:                                  # noqa: BLE001
        return f"探测失败：`{type(e).__name__}: {e}`"
    if not p.get("ok"):
        return f"探测失败：{p.get('reason')}"
    warn = ("\n\n> ⚠️ 满载功耗只有 %.0f / %.0f W —— 这台机器被限功耗了，"
            "真实算力可能是上面的 2 倍以上。请插电并切到「高性能」模式后重测。"
            % (p["max_power_w"], p["power_limit_w"])) if p["throttled"] else ""
    return f"""
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
    with gr.Blocks(title="量化顾问 Agent", css=CSS) as demo:
        gr.Markdown("# Quant Advisor", elem_id="hdr")
        gr.Markdown(
            "给定模型和显存预算，实测每一层能压到几位、会掉多少。"
            "所有数字来自本机测量。", elem_id="hdr")

        # ---- 首屏可见、但不占空间的 API 配置 ----
        with gr.Accordion("🔑 用自己的 API（可选 · 留空则用内置规则）",
                          open=False):
            with gr.Row():
                api_key = gr.Textbox(
                    label="API Key", type="password", scale=4,
                    placeholder="sk-... 不填不影响使用")
                status = gr.Markdown(llm_status(""), scale=2)
            gr.Markdown(
                "填入后由 LLM 把测量结果组织成更自然的表达；"
                "**数值仍全部来自测量管线，且需通过数字校验**。"
                "key 只在本次请求内有效，不写入磁盘。")

        chatbot = gr.Chatbot(
            value=[{"role": "assistant", "content": WELCOME}],
            label="对话", height=460, show_label=False)

        with gr.Row():
            msg = gr.Textbox(
                placeholder="例如：我 8G 显存，想跑 7B 模型",
                show_label=False, scale=8, autofocus=True)
            send = gr.Button("发送", variant="primary", scale=1)

        gr.Examples(examples=EXAMPLES, inputs=msg, label="试试这些")

        # ---- 以下全部折叠，放在下方 ----
        with gr.Accordion("📊 实测数据 · 已知局限", open=False):
            gr.Markdown(DATA_MD)

        with gr.Accordion("🖥️ 检测本机设备（约 20 秒）", open=False):
            dev_btn = gr.Button("开始检测", size="sm")
            dev_out = gr.Markdown("尚未检测。点击上方按钮会实测带宽、算力与功耗。")

        with gr.Accordion("ℹ️ 能力说明", open=False):
            gr.Markdown(SYSTEM_CARD)

        with gr.Row():
            clear = gr.Button("清空对话", size="sm")

        send.click(respond_with_key, [msg, chatbot, api_key], [msg, chatbot])
        msg.submit(respond_with_key, [msg, chatbot, api_key], [msg, chatbot])
        clear.click(lambda: ([], ), None, [chatbot])
        dev_btn.click(device_panel, None, dev_out)
        api_key.change(llm_status, api_key, status)
    return demo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--share", action="store_true", help="生成公网临时链接")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()
    build().launch(share=args.share, server_port=args.port)


if __name__ == "__main__":
    main()

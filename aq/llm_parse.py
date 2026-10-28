"""``aq.llm_parse``：可选的 LLM 意图解析，只解析意图、不生成数字。

分工：

- LLM：把自然语言映射为 ``{intent, model_name, model_b, vram_gib, signal, cand}``
- 工具：所有数字仍来自实测，LLM 不产出任何数值结论

设计原因：规则路由无法覆盖 ``deepseekV3`` / ``R1`` / ``本地那个 30B 的 MoE``
这类写法，而让 LLM 直接作答又会失去「数字可验证」的性质。因此 LLM 只负责
识别意图，回答内容仍由测量管线产出。

配置（任一即可，未配置时自动退回规则路由）：

- 通用：``LLM_API_KEY`` / ``LLM_BASE_URL`` / ``LLM_MODEL``
- 兼容：``DEEPSEEK_API_KEY`` / ``OPENAI_API_KEY`` / ``DEEPSEEK_BASE_URL``
  / ``DSH_LLM_MODEL``

三种配置示例：

1. DeepSeek 官方（成本最低）::

       LLM_API_KEY=sk-xxx
       LLM_BASE_URL=https://api.deepseek.com
       LLM_MODEL=deepseek-chat

2. 硅基流动（国内备用线路）::

       LLM_BASE_URL=https://api.siliconflow.cn/v1
       LLM_MODEL=Qwen/Qwen2.5-7B-Instruct

3. 本地 Ollama（完全离线，不依赖外部网络）::

       ollama run qwen2.5:3b
       LLM_API_KEY=ollama           （占位值即可）
       LLM_BASE_URL=http://localhost:11434/v1
       LLM_MODEL=qwen2.5:3b
"""
from __future__ import annotations

import json
import os
import urllib.request

__all__ = ["available", "llm_parse"]

SYSTEM = """你是意图解析器。只输出 JSON，不要解释。

把用户的话解析成这些字段：
{
  "intent": "fit|sensitivity|plan|table|speed|knowledge|probe|init|onboard",
  "model_name": "模型名，没有就 null",
  "model_b": 数字或 null   （模型总参数，单位十亿）,
  "vram_gib": 数字或 null  （显存，单位 GiB）,
  "signal": "uniform|alphaq|measured|hybrid" (默认 measured),
  "cand": "34|348|2348" (默认 348)
}

intent 判定规则：
- 问能不能装下 / 要多大显存            → fit
- 问哪些层/专家能压、敏感度            → sensitivity
- 要分配方案 / 怎么压                  → plan
- 问每层不同位宽掉多少 / 位宽表        → table
- 问推理速度 / vLLM                    → speed
- 问依据 / 来源 / 可不可靠             → knowledge
- 要求测本机设备                       → probe
- 要实现测本机数据再出方案             → onboard

注意：MoE 模型的 model_b 用【总参数】。
DeepSeek-V3=671, DeepSeek-R1=671, Qwen3-30B-A3B=30.5, Mixtral-8x7B=46.7。"""


def _key() -> str:
    return (os.environ.get("LLM_API_KEY")
            or os.environ.get("DEEPSEEK_API_KEY")
            or os.environ.get("OPENAI_API_KEY") or "")


def _base() -> str:
    return (os.environ.get("LLM_BASE_URL")
            or os.environ.get("DEEPSEEK_BASE_URL")
            or "https://api.deepseek.com")


def _model() -> str:
    return (os.environ.get("LLM_MODEL")
            or os.environ.get("DSH_LLM_MODEL") or "deepseek-chat")


def available() -> bool:
    """判断是否已配置可用的 API key。

    Returns:
        ``True`` 表示检测到 ``LLM_API_KEY`` / ``DEEPSEEK_API_KEY`` /
        ``OPENAI_API_KEY`` 之一；此时 :func:`llm_parse` 才会真正发起请求。
    """
    return bool(_key())


def llm_parse(text: str, timeout: float = 8.0) -> dict | None:
    """把用户输入解析为意图字段字典，无法解析时返回 ``None``。

    请求以 ``temperature=0`` 发出，同一输入的结果应保持一致。返回的字典还要通过
    白名单校验：``intent`` 必须属于已知集合，否则同样返回 ``None``，由调用方退回
    规则路由。本函数是同步阻塞调用，一次请求耗时取决于 ``timeout``。

    Args:
        text: 用户原始输入，不做预处理，直接作为 user message 的内容。
        timeout: 单次 HTTP 请求超时，单位秒。默认 8.0。

    Returns:
        解析结果字典，可能包含 ``intent`` / ``model_name`` / ``model_b`` /
        ``vram_gib`` / ``signal`` / ``cand``。以下情况返回 ``None``：未配置
        API key、请求或 JSON 解析失败、``intent`` 不在白名单内。

    Note:
        LLM 只负责意图识别，所有数值结论仍由实测管线给出。
    """
    key = _key()
    if not key:
        return None
    base = _base()
    model = _model()
    url = base.rstrip("/") + "/chat/completions"
    body = {
        "model": model,
        "messages": [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": text}],
        "temperature": 0,
        "max_tokens": 200,
        "response_format": {"type": "json_object"},
    }
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            out = json.loads(r.read().decode())
        content = out["choices"][0]["message"]["content"]
        d = json.loads(content)
        # 白名单校验：模型可能返回训练时见过的其它 intent 取值，非白名单一律
        # 视为解析失败，避免下游分支收到未定义的行为。
        ok_intents = {"fit", "sensitivity", "plan", "table", "speed",
                      "knowledge", "probe", "init", "onboard"}
        if d.get("intent") not in ok_intents:
            return None
        return d
    # NOTE: 这里刻意捕获所有异常（连接超时、HTTP 非 2xx、响应非 JSON、字段
    # 缺失），统一转为 None，由调用方退回规则路由；任何解析失败都不应中断主流程。
    except Exception:                                      # noqa: BLE001
        return None

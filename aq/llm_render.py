"""表达层：用 LLM 把工具产出的结构化事实组织成自然语言。

设计原则
  测量层负责事实，表达层负责措辞。两层严格分离：
  · LLM 只拿到【工具已算出的文本】，不接触模型权重、不执行任何计算
  · 输出后做数字校验：出现的每个数值必须能在事实文本中找到
  · 校验不通过、超时、无 key —— 一律返回 ``None``，调用方退回模板

这样做的原因
  直接用 LLM 回答会引入不可验证的数字，而本工具的核心主张是「每个数字可追溯」。
  把 LLM 限制在「重组已有事实」上，既获得自然表达，又不牺牲可审计性。
"""
from __future__ import annotations

import os
import re
import urllib.request

__all__ = ["available", "render", "check_numbers"]


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
    """表达层是否可用（需要配置 API key）。"""
    return bool(_key())


# 小于此值的整数视为「通用词」，如「两种方案」「三层」，不参与校验
_SMALL_INT_MAX = 12

_NUM_RE = re.compile(r"\d+(?:\.\d+)?")


def _numbers(text: str) -> set[str]:
    return set(_NUM_RE.findall(text))


def check_numbers(output: str, facts: str) -> tuple[bool, list[str]]:
    """校验 LLM 输出没有引入事实之外的数字。

    Args:
        output: LLM 生成的文本。
        facts: 工具产出的原始文本（唯一允许的数字来源）。

    Returns:
        ``(是否通过, 违规数字列表)``。小数必须精确出现；
        绝对值 ≤ 12 的整数视为通用词而放行。
    """
    allowed = _numbers(facts)
    bad = []
    for tok in _numbers(output):
        if tok in allowed:
            continue
        if "." not in tok and abs(int(tok)) <= _SMALL_INT_MAX:
            continue
        bad.append(tok)
    return (not bad), bad


_SYSTEM = """你是技术助手的表达层，负责把结构化事实讲清楚。

硬约束（违反即失败）：
1. 只能使用【事实】中出现的数值。不得推算、换算、四舍五入或引入任何新数字。
2. 事实中没有的内容一律不写，不补充常识、不猜测。
3. 保留原文单位（GiB / bit / % / tok/s / 秒）。
4. 不要复述「事实」二字的标签，直接回答用户。
5. 中文回答，先给结论，再给依据，控制在 400 字内。
6. 保留原文中的「实测 / 估计 / 预测」标记，不得改写其含义。"""


def render(question: str, facts: str, timeout: float = 20.0,
           max_tokens: int = 700) -> str | None:
    """用 LLM 把事实文本组织成回答。

    Args:
        question: 用户的原始问题。
        facts: 工具产出的完整文本，是唯一允许的数字来源。
        timeout: 请求超时，单位秒。
        max_tokens: 输出长度上限。

    Returns:
        组织后的回答；无 key、请求失败或数字校验不通过时返回 ``None``，
        调用方应当退回模板输出。
    """
    key = _key()
    if not key:
        return None

    url = _base().rstrip("/") + "/chat/completions"
    body = {
        "model": _model(),
        "messages": [
            {"role": "system", "content": _SYSTEM},
            {"role": "user",
             "content": f"【事实】\n{facts}\n\n【用户问题】\n{question}"},
        ],
        "temperature": 0.2,
        "max_tokens": max_tokens,
    }
    import json
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            out = json.loads(r.read().decode())
        text = out["choices"][0]["message"]["content"].strip()
    except Exception:                                       # noqa: BLE001
        return None

    if not text:
        return None
    ok, bad = check_numbers(text, facts)
    if not ok:
        # 引入了事实之外的数字，判定为不可信，退回模板
        return None
    return text

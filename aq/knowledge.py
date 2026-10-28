"""项目事实库：检索、引用与矛盾列举。

每条事实是一个 claim 记录，含条件、证据、置信度，以及指向其他记录的矛盾边。
散文文档中的事实无法被机器检索，也无法标注来源是实测还是转述，故改用结构化记录。

公开 API：

- :func:`query`：按关键词与标签检索 claim
- :func:`cite`：生成引用块，供回答末尾附带
- :func:`conflicts`：列出所有互相矛盾的 claim 对
- :func:`stats`：按置信度与主题统计 claim 数量

设计取舍：

- 存储用 JSON 而非 YAML，不引入第三方依赖。
- ``confidence`` 仅三档：``high``（实测）/ ``medium``（外部来源或单次测量）/
  ``low``（转述未验证）。
- 矛盾不隐藏，显式并列：说明不一致的原因比假定一致更可信。

已知局限：检索为词面匹配，不做语义相似度；中文按二元组切分，
同义改写会漏召回。
"""
from __future__ import annotations

import json
import re
from pathlib import Path

__all__ = ["load", "query", "cite", "conflicts", "all_ids", "stats"]

_DEFAULT = Path("knowledge/claims.json")
_CACHE: dict | None = None


def load(path: str | Path = _DEFAULT) -> dict:
    """加载事实库，默认路径的结果会被缓存。

    Args:
        path: JSON 文件路径。

    Returns:
        ``{"meta": {...}, "claims": [...]}``；文件不存在时返回空结构。
    """
    global _CACHE
    if _CACHE is not None and path == _DEFAULT:
        return _CACHE
    p = Path(path)
    if not p.exists():
        return {"meta": {}, "claims": []}
    data = json.loads(p.read_text(encoding="utf-8"))
    if path == _DEFAULT:
        _CACHE = data
    return data


def all_ids() -> list[str]:
    """列出所有事实条目的 id。

    Returns:
        id 列表，顺序与文件中的定义顺序一致。
    """
    return [c["id"] for c in load()["claims"]]


def stats() -> dict:
    """汇总事实库的规模与分布。

    Returns:
        含 ``total`` / ``by_confidence`` / ``by_topic`` 三个键的字典。
    """
    cs = load()["claims"]
    out = {"total": len(cs), "by_confidence": {}, "by_topic": {}}
    for c in cs:
        out["by_confidence"][c.get("confidence", "?")] = \
            out["by_confidence"].get(c.get("confidence", "?"), 0) + 1
        out["by_topic"][c.get("topic", "?")] = \
            out["by_topic"].get(c.get("topic", "?"), 0) + 1
    return out


def _tokens(text: str) -> set[str]:
    """切分为检索用的 token 集合：英文词 + 中文二元组。"""
    t = text.lower()
    en = set(re.findall(r"[a-z0-9_]{2,}", t))
    cjk = re.findall(r"[\u4e00-\u9fff]+", t)
    zh = set()
    for seg in cjk:
        for i in range(len(seg) - 1):
            zh.add(seg[i:i + 2])
        zh.add(seg)
    return en | zh


def query(text: str, top: int = 5) -> list[dict]:
    """按关键词检索 claim，返回得分最高的若干条。

    得分 = 命中 token 数之和，标签命中按 2 倍加权。参与匹配的字段为
    ``claim`` / ``conditions`` / ``evidence`` / ``tags`` / ``topic``。

    Args:
        text: 查询串，英文取 ``[a-z0-9_]{2,}`` 的整词，中文取二元组。
        top: 返回条数上限，默认 5。

    Returns:
        按得分降序排列的 claim 列表，最多 ``top`` 条；查询串无有效 token
        或全部未命中时返回空列表。

    Note:
        复杂度 O(n·m)，n 为 claim 数，m 为单条 claim 的平均字段长度——
        每条候选在每次查询时都会重新分词。
    """
    qs = _tokens(text)
    if not qs:
        return []
    scored = []
    for c in load()["claims"]:
        hay = _tokens(" ".join([
            c["claim"], c.get("conditions", ""), c.get("evidence", ""),
            " ".join(c.get("tags", [])), c.get("topic", ""),
        ]))
        hit = len(qs & hay)
        if hit:
            # 标签是人工提炼的检索入口，区分度高于正文，故按 2 倍权重计入。
            tagh = len(qs & _tokens(" ".join(c.get("tags", []))))
            scored.append((hit + 2 * tagh, c))
    scored.sort(key=lambda x: -x[0])
    return [c for _, c in scored[:top]]


def get(cid: str) -> dict | None:
    """按 id 取单条事实。

    Args:
        cid: 事实 id，例如 ``"K016"``。

    Returns:
        事实字典；不存在时返回 ``None``。
    """
    for c in load()["claims"]:
        if c["id"] == cid:
            return c
    return None


def conflicts() -> list[tuple[dict, dict]]:
    """列出所有显式标注为矛盾的 claim 对。

    双向矛盾边只计一对；指向不存在 id 的边被忽略。

    Returns:
        ``(claim_a, claim_b)`` 元组列表，顺序与 ``contradicts`` 声明顺序一致。
    """
    seen, out = set(), []
    for c in load()["claims"]:
        for oid in c.get("contradicts", []):
            key = tuple(sorted([c["id"], oid]))
            if key in seen:
                continue
            seen.add(key)
            o = get(oid)
            if o:
                out.append((c, o))
    return out


_CONF = {"high": "实测", "medium": "外部/单次", "low": "转述未验证"}


def cite(ids: list[str], show_evidence: bool = True) -> str:
    """把 claim id 渲染为 Markdown 引用块。

    Args:
        ids: 要引用的 claim id 列表；不存在的 id 直接跳过。
        show_evidence: 是否附带 ``evidence`` 对应的「依据」行。

    Returns:
        Markdown 文本，以 ``### 依据（可查证）`` 起始；``ids`` 全部无效时
        返回空字符串。
    """
    rows = []
    for cid in ids:
        c = get(cid)
        if not c:
            continue
        line = (f"**{c['id']}**（{_CONF.get(c.get('confidence'), '?')}）"
                f"{c['claim']}")
        if show_evidence and c.get("evidence"):
            line += f"\n  - 依据：{c['evidence']}"
        if c.get("conditions"):
            line += f"\n  - 条件：{c['conditions']}"
        rows.append(line)
    if not rows:
        return ""
    return "### 依据（可查证）\n\n" + "\n\n".join(rows) + "\n"


def conflicts_block() -> str:
    """把矛盾清单渲染为 Markdown 块。

    Returns:
        Markdown 文本；无矛盾时返回空字符串。
    """
    cs = conflicts()
    if not cs:
        return ""
    out = ["### ⚠️ 我记录到的矛盾（不隐藏）\n"]
    for a, b in cs:
        out.append(f"- **{a['id']}**（{_CONF.get(a.get('confidence'))}）：{a['claim']}")
        out.append(f"  - 条件：{a.get('conditions','')}")
        out.append(f"- **{b['id']}**（{_CONF.get(b.get('confidence'))}）：{b['claim']}")
        out.append(f"  - 条件：{b.get('conditions','')}")
        out.append("  - **两者可能都对，取决于环境。** 这正是「必须在你自己的机器上测」的理由。\n")
    return "\n".join(out)

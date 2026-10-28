"""自适应混合精度量化工具包。

给定模型与显存预算，输出逐层位宽分配方案、Pareto 前沿，以及显存到卡型
再到年成本的换算报告。

模块一览
--------
    aq.metrics       PPL 评测（主指标）
    aq.bench         速度测量（TTFT / TPOT / 吞吐）
    aq.fakequant     伪量化（逐层敏感度分析的载体）
    aq.sensitivity   逐层敏感度扫描
    aq.allocate      位宽分配（约束优化）
    aq.pareto        Pareto 前沿生成
    aq.cost          显存 → 卡型 → 年成本

设计原则
--------
1. 每个模块可独立 import 与测试，避免模块之间的隐式耦合。
2. 模块之间不互相 import 具体模型类，只收发 numpy / torch / csv / dict。
3. 「给定模型」相关的形状参数从 config 读取，不硬编码具体架构，
   否则无法适配任意模型输入。

实现状态
--------
``aq.cost`` 已实现，为纯计算模块，可独立测试。

.. warning::
    除 ``aq.cost`` 外，其余模块当前仅有骨架或占位实现，公开 API 尚不稳定，
    可能在不另行通知的情况下变更。
"""

__version__ = "0.1.0"

# 只导出稳定的公共 API。模块内部函数请显式 `from aq.xxx import yyy`。
from aq import cost  # noqa: F401  （当前唯一已实现且稳定的模块）

__all__ = ["cost", "__version__"]

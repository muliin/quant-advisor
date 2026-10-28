# 第三方声明

本仓库**不包含**任何第三方源代码副本、模型权重或数据集。
下列条目说明所引用工作的许可状态与归属。

---

## 一、被分析/对照的外部工作

### AlphaQ

- **论文**：*AlphaQ: Calibration-free Bit Allocation for Mixture-of-Experts Quantization*，
  arXiv:2606.04980，NeurIPS 2026
- **官方代码仓库**：**未提供 LICENSE 文件**
- **本仓库的处理**：
  - **未复制、未改编其任何源代码**
  - `aq/spectral.py` 中的谱指标（Hill 估计、η 代价函数、γ 的计算）由作者**依据论文描述独立实现**
  - `aq/solve.py` 中的 MCKP 求解为本项目自行实现的动态规划，
    未使用 AlphaQ 代码中的 `gurobipy` / PuLP 求解器
  - 论文陈述与实现的差异记录在 `00_总纲/AlphaQ-对照分析.md` 中，
    作为对照分析而非代码衍生
- **引用建议**：

  ```bibtex
  @inproceedings{alphaq2026,
    title     = {AlphaQ: Calibration-free Bit Allocation for Mixture-of-Experts Quantization},
    booktitle = {NeurIPS},
    year      = {2026},
    note      = {arXiv:2606.04980}
  }
  ```

> **说明**：因官方仓库缺少 LICENSE，本项目采取「只读论文、不读代码」的策略以避免许可风险。
> 如该仓库后续补充许可证，本声明应随之更新。

### unsloth-zoo

- 仅引用其 issue / PR 标题作为**外部旁证**
  （`Fix OLMoE load_in_4bit crash: route fused experts through the MoE backend`，PR #915）
- 未使用其代码

---

## 二、运行时依赖

本仓库通过 `requirements.txt` 声明依赖，**不打包**其源码：

| 依赖 | 许可证 |
|---|---|
| PyTorch | BSD-3-Clause |
| transformers | Apache-2.0 |
| bitsandbytes | MIT |
| accelerate | Apache-2.0 |
| numpy | BSD-3-Clause |
| gradio | Apache-2.0 |
| matplotlib | PSF-based (matplotlib license) |

安装与分发这些依赖时，请遵守各自的许可证。

---

## 三、被测量的模型

本仓库**不重新分发任何模型权重**。所有实验通过 Hugging Face Hub 拉取：

| 模型 | 许可证 | 用途 |
|---|---|---|
| `Qwen/Qwen2.5-1.5B-Instruct` | Apache-2.0 | dense 基准（196 个模块）|
| `allenai/OLMoE-1B-7B-0924` | Apache-2.0 | MoE 基准（2048 个专家决策点）|

**导出的量化产物（`out/`）属于上述模型的衍生作品，其分发须遵守原模型许可证。**
本仓库的 `.gitignore` 已排除 `out/` 与 `.models/`，不随仓库分发。

---

## 四、评测文本

`data/` 下的英文/中文评测文本由本项目自行编写，
用于 PPL 测量，不来自任何受版权保护的数据集。

---

## 五、引用本项目

```bibtex
@misc{quantadvisor2026,
  title  = {Quant Advisor: A Measurement-First Quantization Agent for MoE Models},
  author = {quant-advisor contributors},
  year   = {2026},
  note   = {Zhihu Hackathon / Software Elite Innovation Competition}
}
```

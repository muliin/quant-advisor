# Quant Advisor

**给定一个模型和显存预算，实测它每一层能压到几位、会掉多少，然后真的导出一个变小的模型。**

> 通用助手会说「用 4bit，省 75% 显存」。
> 但它不知道你这一层压下去会掉多少 —— 这个工具会在你的模型上逐单元测出来。

面向 MoE（混合专家）模型设计，也支持 dense 模型和任意 HuggingFace 模型。

---

## 它解决什么问题

量化工具通常只回答「压到几位」。真正难的是另外三件事：

1. **哪些部分不能压？** —— 不同层的敏感度差异可达 3 倍以上
2. **压下去会掉多少？** —— 需要实测，不能靠经验
3. **最终能交付什么？** —— 一个能加载的、真的变小的模型

本工具把它们串成一条可复现的链：

```
结构发现 → 逐单元敏感度扫描 → 位宽分配 → 量化导出 → 加载验证
```

---

## 三条命令

```bash
# ① 问一句话（意图路由 + 实测数据）
python -m aq.ask "8G 显存能跑 Qwen/Qwen2.5-1.5B-Instruct 吗"

# ② 在任意模型上跑完整管线（结构 → 扫描 → 方案 → 报告）
python -m aq.onboard --model Qwen/Qwen2.5-1.5B-Instruct --bits 4,8 --cand 3,4,8

# ③ 量化并导出（产物可用 load_in_4bit 加载）
python -m aq.export --model Qwen/Qwen2.5-1.5B-Instruct --out out/qwen4bit
```

也可以用网页界面：

```bash
python app.py
```

---

## 实测结果

以下数字均在本项目环境中实测，脚本与数据见 `results/`，可复现。

### MoE（OLMoE-1B-7B，16 层 × 64 专家 × 2 张量 = 2048 个决策点）

| 结论 | 数字 |
|---|---|
| **2bit 是悬崖** | 每一层都掉 **+3.3% ~ +17.4%** |
| **3bit 几乎无损** | **−0.6% ~ +1.8%**（相对 bf16 为 5.3× 压缩）|
| 层敏感度趋势 | 仅 2bit 存在（ρ=+0.76）；3bit 消失（ρ=+0.16）|
| **同预算下分配方向的影响** | **41.7%**（+7.40% vs +12.69%，平均位宽均为 3.5）|
| 加法代价模型的偏差 | 预测 +3.50%，实测 **+7.21%**（低估 2.06×）|
| α 谱指标的判别力 | 0.15（"第几层"这个整数为 0.52）|

### 推理加速

| 配置 | 吞吐 | 环境 |
|---|---|---|
| bnb 4bit + PyTorch eager | 26.3 tok/s | RTX 3090 |
| bf16 + vLLM | 203.1 tok/s | RTX 3090 |
| AWQ-INT4 + vLLM | **364.8 tok/s** | RTX 3090 |

**bnb 的 4bit 只用到 9.9% 显存带宽**（RTX 5070 Laptop）—— 它省显存但不提速。
换推理后端的收益（13.9×）远大于换精度（1.80×）。

---

## 设计原则

### ① 测量优先

每个数字都标注来源：**实测**（有 CSV）/ **估计**（按参数量推算）/ **预测**（加法代价模型，仅用于排序）。

工具会主动提示：**自身的加法代价模型会低估 1~2 倍，且方向不一定**。

### ② LLM 只做表达，不产出数字

可选 LLM 层承担两件事：把自然语言解析成 `{intent, model_b, vram_gib}`，
以及把测量结果组织成通顺措辞。

**LLM 拿不到模型权重，输出还要经过数字校验** ——
回答中出现的每个数值必须能在测量文本中找到，否则整体作废、退回模板。

未配置 API key 时自动退回规则路由与模板，**离线可用**。
接入方法见 [LLM_SETUP.md](LLM_SETUP.md)。

### ③ 边界明确

- 伪量化必须在 bf16 权重上进行 → 模型装不进 bf16 时无法测敏感度
- bnb 在部分 transformers 版本下**不量化融合专家张量** → MoE 模型导出受限
- vLLM 不支持逐层/逐专家位宽 → 可部署的混合精度形式为「NF4 + 若干层 bf16」

---

## 项目结构

```
aq/                     核心库（19 个模块，约 4400 行）
  onboard.py            任意模型一条龙：describe / scan / plan / report
  export.py             真量化导出 + 可加载验证
  sensitivity.py        敏感度扫描内核（伪量化 + 精确还原）
  fakequant.py          NF4 量化原语（量化 → 反量化）
  solve.py              位宽分配的动态规划求解器
  allocate.py           量化单元建模与贪心基线
  spectral.py           谱指标 α 与代价函数 η（AlphaQ 论文的独立实现）
  agent.py              意图路由 + 回答渲染 + 设备探测
  models.py             20 个常见模型的知识表
  knowledge.py          24 条实测事实的结构化知识库
  llm_parse.py          可选的 LLM 意图解析层
  ask.py                命令行问答入口（Skill 主命令）
  bench.py / metrics.py 性能与 PPL 基准
  cost.py / pareto.py   成本模型与 Pareto 前沿

app.py                  Gradio 网页界面
scripts/                生成 results/ 的扫描脚本（可复现）
tools/verify_aq.py      求解器自检
skills/quant-advisor/   Agent 能力包（SKILL.md）
knowledge/claims.json   知识库数据（运行时读取）
data/                   评测文本 en_eval.txt / zh_eval.txt
results/                实测 CSV 与实验报告（运行时读取）
```

---

## 安装

```bash
pip install -r requirements.txt
```

需要 CUDA GPU。模型通过 Hugging Face Hub 下载，国内可设置镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

---

## 技术文档

| 文档 | 内容 |
|---|---|
| [`results/MoE逐专家扫描-实测结果.md`](results/MoE逐专家扫描-实测结果.md) | 2048 个专家决策点 + 层×位宽 96 点 + 分配受控实验的完整记录 |
| [`results/Qwen2.5-1.5B-Instruct_report.md`](results/Qwen2.5-1.5B-Instruct_report.md) | 任意模型管线的示例输出 |
| [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) | 第三方工作与许可状态 |

---

## 已知限制

| 限制 | 原因 |
|---|---|
| 单模型、单评测文本 | 绝对 ΔPPL 随文本变化；排序应稳定 |
| 伪量化 ≠ 真实部署 | 测的是量化误差，不含 kernel 实现差异 |
| MoE 导出受限 | bnb 在当前版本不量化融合专家张量 |
| 混合位宽无法在 vLLM 运行 | vLLM 的量化是全局配置 |
| 单卡、单次测量 | 未做多次重复与置信区间 |

---

## 许可

[MIT](LICENSE)

第三方工作（AlphaQ 等）的引用与许可状态见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
本仓库**不重新分发任何模型权重**。

---

## 引用

```bibtex
@misc{quantadvisor2026,
  title  = {Quant Advisor: A Measurement-First Quantization Agent for MoE Models},
  author = {quant-advisor contributors},
  year   = {2026}
}
```

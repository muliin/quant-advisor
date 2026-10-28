---
name: quant-advisor
description: 给定量化目标（模型 + 显存预算），实测该模型每个可量化单元的敏感度，推荐若干位宽分配方案，并把选定方案真正导出成可加载的量化模型。所有数值来自本机实测，不依赖训练数据或直觉。
when_to_use: 用户问「这个模型能不能在我的显卡上跑」「怎么把它压小」「量化我的模型」「哪些层不能压到 4bit」「MoE 的专家能不能压」时使用。
version: 1.0
---

# quant-advisor

**一句话**：**别猜了，测。** 给模型和显存，我告诉你哪些层能压、压到几位、能省多少，并把结果真的导出成一个能加载的模型。

---

## 我解决什么问题

通用助手会说：「用 4bit 加载即可，大概省 75% 显存。」
**但它不知道你这一层压下去会掉多少。**

我会在**你的模型**上逐单元实测，然后给出方案。每个数字都能追到一个 CSV。

---

## 三种调用方式

### ① 直接问（最常用）

```bash
python -m aq.ask "我 8G 显存，想跑 Qwen/Qwen2.5-1.5B-Instruct"
```

内部按意图路由到对应工具，输出 markdown 回答（带「实测/估计」标记）。

### ② 在任意模型上跑完整管线

```bash
# 结构发现 → 逐单元 × 位宽扫描 → 三方案 → 报告
python -m aq.onboard --model Qwen/Qwen2.5-1.5B-Instruct --bits 4,8 --cand 3,4,8

# 只扫一部分（快速验证管线）
python -m aq.onboard --model /path/to/model --bits 4 --limit 20
```

产出：`results/<model>_scan.csv`、`results/<model>_report.md`

### ③ 量化并导出

```bash
# 均匀 4bit
python -m aq.export --model Qwen/Qwen2.5-1.5B-Instruct --out out/qwen4bit

# ★ 可部署的混合精度：整体 NF4，敏感层留 bf16
python -m aq.export --model Qwen/Qwen2.5-1.5B-Instruct --out out/qwen4bit_skip \
  --skip model.layers.27.mlp.down_proj,model.layers.26.mlp.down_proj
```

导出的目录用 `load_in_4bit=True` 可直接加载。

---

## 输出怎么读

| 标记 | 含义 |
|---|---|
| **实测** | 在该模型/该机器上真跑过，有 CSV |
| **估计** | 按参数量和位宽算的，没有跑 |
| **预测 ΔPPL** | 加法模型推算，**误差 1~2 倍、方向不定**，只能用来排序 |

### 三条硬约束（必须主动告知用户）

1. **伪量化必须在 bf16 权重上做** → 模型装不进 bf16 就无法测敏感度
2. **bnb 不量化融合专家张量**（OLMoE 实测：仍占 12.5 GiB）→ MoE 模型导不出来
3. **vLLM 不支持逐层/逐专家位宽** → 可部署的混合精度只能是「NF4 + 若干层 bf16」

---

## 我实测出来的关键结论（可直接引用）

| 结论 | 数字 | 适用条件 |
|---|---|---|
| **2bit 是悬崖** | 每层 +3.3% ~ +17.4% | OLMoE-1B-7B，16 层 |
| **3bit 几乎无损** | −0.6% ~ +1.8%（5.3× 压缩）| 同上 |
| **3bit 以上层趋势消失** | ρ: 2bit +0.76 → 3bit +0.16 | 同上 |
| **同预算分配方向影响** | **41.7%**（+7.40% vs +12.69%）| 平均 3.5 bit |
| **加法预测会骗优化器** | 预测 +3.50%，实测 +7.21%（2.06×）| DP 择优方案 |
| **bnb 4bit 不提速** | 带宽利用仅 **9.9%** | RTX 5070 Laptop |
| **换后端 > 换精度** | 13.9× vs 1.80× | vLLM vs bnb eager |

**这些不是文献值，是本项目在真机上测的**（24 条事实见 `knowledge/claims.json`）。

---

## 我不做什么

- **不预测我没测过的模型** —— 会说「需要在你模型上跑一次，约 2 分钟」
- **不把加法预测当结论** —— 只用于排序，最终必须实测
- **不承诺 4bit 能提速** —— 实测只用 9.9% 带宽，提速靠后端

---

## 依赖

`torch`（CUDA）、`transformers`、`bitsandbytes`。不需要网络（模型可本地）。

Python 入口：`aq/` 包（17 个模块）。

| 模块 | 作用 |
|---|---|
| `aq.onboard` | 结构发现 / 扫描 / 方案 / 报告 |
| `aq.export` | 真量化导出 + 可加载验证 |
| `aq.agent` | 意图路由 + 回答渲染 |
| `aq.models` | 常见模型知识表（20 个） |
| `aq.knowledge` | 24 条事实库（含矛盾并列） |
| `aq.solve` | DP 求位宽分配 |
| `aq.sensitivity` / `aq.fakequant` | 扫描内核与伪量化 |

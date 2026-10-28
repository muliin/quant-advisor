# 接入 LLM（可选）

**本项目不依赖 LLM 也能完整运行。** 不配置任何 key 时，Agent 使用规则路由 + 模板回答。

配置之后，LLM 承担两件事：

| 做什么 | 不做什么 |
|---|---|
| 把自然语言问题解析成 `{意图, 模型规模, 显存}` | **不产出任何数值结论** |
| 把工具返回的结构化事实组织成通顺的中文 | **不接触模型权重、不执行计算** |

---

## 为什么这样分工

直接用 LLM 回答会引入无法验证的数字（"约省 75% 显存"），而本工具的核心主张是**每个数字都能追到 CSV**。

因此把 LLM 限制在「重组已有事实」上：

```
用户问题
   ↓
LLM 解析意图  ──→  {intent, model_b, vram_gib}
   ↓
测量管线     ──→  查表 / 扫描 / DP 求解  ← 所有数字在这里产生
   ↓
LLM 组织措辞 ──→  只能使用上一步文本中出现的数字
   ↓
数字校验     ──→  出现新数字则整体作废，退回模板
```

**最后一步是硬防线**，实现在 `aq/llm_render.py::check_numbers`：

```python
>>> check_numbers("压缩率 75%", facts="3bit 压缩 5.3 倍")
(False, ['75'])          # 判定不可信，退回模板
```

---

## 三种后端

### ① DeepSeek（成本最低）

在 https://platform.deepseek.com 创建 API key，然后：

```bash
# Linux / macOS
export LLM_API_KEY="sk-..."
export LLM_BASE_URL="https://api.deepseek.com"
export LLM_MODEL="deepseek-chat"
```

```powershell
# Windows PowerShell
$env:LLM_API_KEY  = "sk-..."
$env:LLM_BASE_URL = "https://api.deepseek.com"
$env:LLM_MODEL    = "deepseek-chat"
```

### ② 硅基流动（国内备用线路）

```bash
export LLM_API_KEY="sk-..."
export LLM_BASE_URL="https://api.siliconflow.cn/v1"
export LLM_MODEL="Qwen/Qwen2.5-7B-Instruct"
```

### ③ 本地 Ollama（完全离线）

```bash
ollama pull qwen2.5:3b
ollama serve
```

```powershell
$env:LLM_API_KEY  = "ollama"                       # 占位值，Ollama 不校验
$env:LLM_BASE_URL = "http://localhost:11434/v1"
$env:LLM_MODEL    = "qwen2.5:3b"
```

---

## 用 `.env` 文件（推荐本地开发）

```bash
cp .env.example .env
# 编辑 .env，填入真实 key
```

`.env` 已在 `.gitignore` 中，不会被提交。

如果想自动加载，可以用 `python-dotenv`（非必需依赖）：

```python
from dotenv import load_dotenv
load_dotenv()
```

---

## 在网页界面里临时覆盖

`python app.py` 启动后，右侧栏底部有一个 **API Key（可选）** 输入框：

- **留空** → 使用环境变量里的 key；环境变量也没有 → 规则模式
- **填入** → 仅覆盖本次请求，用完立即恢复原值（不写文件、不写日志）

这解决了公开体验链接的问题：**部署者放一把 key 让访客开箱可用，访客也可以填自己的 key 而不消耗部署者的额度。**

---

## 验证是否接通

```bash
python -m aq.ask "llm 自检"
```

配置正确时输出：

```
✅ LLM 解析正常（0.83s）
输入：我16G显存，想本地部署deepseekV3模型
解析结果：{"intent":"fit","model_b":671,"vram_gib":16,...}
```

未配置时输出配置指引；key 无效或网络不通时提示检查项。

---

## 环境变量清单

| 变量 | 默认值 | 说明 |
|---|---|---|
| `LLM_API_KEY` | 空 | 主变量。也接受 `DEEPSEEK_API_KEY` / `OPENAI_API_KEY` |
| `LLM_BASE_URL` | `https://api.deepseek.com` | 兼容任何 OpenAI 格式的 `/chat/completions` |
| `LLM_MODEL` | `deepseek-chat` | 模型名 |

---

## 安全须知

- **永远不要把 key 写进代码或提交到仓库。** 本项目只从环境变量与界面输入读取。
- key 泄露后到对应平台**立即吊销**。
- 如果误提交过 key，仅删除文件不够 —— git 历史里仍有，需要重写历史并吊销 key。
- 部署到公开平台时，用平台的 Secret / 环境变量功能，不要写进 `app.py`。

---

## 降级行为

| 情况 | 行为 |
|---|---|
| 未配置 key | 规则路由 + 模板回答，**全部功能可用** |
| key 无效 / 超时（8 s 解析、20 s 组织） | 自动退回规则与模板 |
| LLM 输出引入新数字 | 校验失败，退回模板 |
| 网络中断 | 同上，**离线可用** |

**设计意图：LLM 是增强项，不是必需品。** 任何一处失败都不应让工具不可用。

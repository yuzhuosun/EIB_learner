# EIB_learner

## 项目简介

该仓库实现了论文 **"Understanding the Information Propagation Effects of Communication Topologies in LLM-based Multi-Agent Systems" (EMNLP 2025 main)** 的核心实验代码。整体框架继承自 GDesigner，EIB learner 的图融合逻辑在 `GDesigner/graph/graph.py` 中，实验入口在 `experiments/` 目录。

---

## 1. 本地复现前提

- Python 版本建议：**3.10 ~ 3.11**
- 需要可用的 OpenAI 兼容 API（`BASE_URL` + `API_KEY`）
- 建议在 Linux / macOS 或 WSL 环境复现

---

## 2. 本地环境搭建（推荐完整命令）

```bash
# 1) 克隆仓库
git clone <your-repo-url>
cd EIB_learner

# 2) 创建虚拟环境
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\\Scripts\\activate

# 3) 安装依赖
pip install -U pip
pip install -r requirements.txt
```

---

## 3. 配置 API 环境变量

复制模板并填写：

```bash
cp template.env .env
```

在 `.env` 中配置：

```bash
BASE_URL="..."   # OpenAI 兼容后端地址
API_KEY="..."    # 你的 API Key
```

> 注意：代码通过 `GDesigner/llm/gpt_chat.py` 读取环境变量并发起请求；若未配置会在调用模型时失败。

---

## 4. 数据准备

### MMLU

`run_mmlu.py` 会使用：
- `MMLUDataset('dev')` 作为训练集
- `MMLUDataset('val')` 作为验证集

并依赖 `datasets/MMLU/download.py` 获取数据。首次运行前，建议先执行：

```bash
python datasets/MMLU/download.py
```

### GSM8K / HumanEval

仓库已包含示例数据文件：
- `datasets/gsm8k/gsm8k.jsonl`
- `datasets/humaneval/humaneval-py.jsonl`

可直接运行对应脚本（见第 7 节）。

---

## 5. 最小可运行验证（建议先跑）

先用最小配置确认“代码 + API + 数据”链路打通：

```bash
python experiments/run_mmlu.py \
  --mode FullConnected \
  --batch_size 1 \
  --agent_nums 2 \
  --num_iterations 1 \
  --num_rounds 1
```

如果该命令能正常输出 `Score: ...`，说明本地基础复现成功。

---

## 6. 论文主流程（MMLU）

```bash
python experiments/run_mmlu.py \
  --mode FullConnected \
  --batch_size 4 \
  --agent_nums 6 \
  --num_iterations 10 \
  --num_rounds 1 \
  --optimized_spatial
```

### 常用参数解释

- `--mode`：图拓扑（如 `FullConnected` / `Random` / `Chain` / `Star` / `Mesh`）
- `--agent_names` + `--agent_nums`：Agent 类型及数量（两者长度必须一致）
- `--optimized_spatial`：开启空间结构优化
- `--optimized_temporal`：开启时间结构优化
- `--llm_name`：底层模型名（默认 `gpt-4o`）

---

## 7. 其他实验入口

```bash
# GSM8K
python experiments/run_gsm8k.py

# HumanEval
python experiments/run_humaneval.py

# MMLU 训练与评估拆分脚本
python experiments/train_mmlu.py
python experiments/evaluate_mmlu.py
```

---

## 8. 你可能需要修改的代码位置（最关键）

### A) 切换模型或推理后端

- 文件：`GDesigner/llm/gpt_chat.py`
- 典型修改：模型名、请求参数（温度、token 限制、重试策略等）

### B) 调整图拓扑构造逻辑

- 文件：`experiments/run_mmlu.py` 中 `get_kwargs(...)`
- 典型修改：
  - 新增自定义拓扑模式
  - 修改 `fixed_spatial_masks` / `fixed_temporal_masks` 生成规则

### C) 调整 Agent 角色行为

- 目录：`GDesigner/agents/`
- 典型修改：
  - `analyze_agent.py`
  - `adversarial_agent.py`
  - `final_decision.py`

### D) 修复优化阶段模型加载（非常重要）

当前 `run_mmlu.py` 在训练后包含如下占位路径：

- `gcn_model_path = ""`
- `mlp_mode_path = ""`
- `gcn_chain_path = ""`
- `mlp_chain_path = ""`
- `mlp_fuse_path = ""`

若你开启 `--optimized_spatial` / `--optimized_temporal`，需要把这些路径改成真实 checkpoint 文件，否则会在 `torch.load(...)` 报错。

---

## 9. 常见报错与排查

1. **401/鉴权失败**：检查 `.env` 的 `BASE_URL` 与 `API_KEY`。
2. **数据集读取错误**：先执行 `python datasets/MMLU/download.py`。
3. **CUDA/torch 相关错误**：先在 CPU 环境跑通最小命令，再逐步开启更大 batch。
4. **优化阶段加载失败**：按第 8-D 节填写 checkpoint 路径。

---

## 10. 结果复现实用建议

- 先固定随机种子并记录每次参数。
- 每种拓扑（如 `FullConnected`、`Chain`、`Star`）至少重复 3 次。
- 保存每次实验命令、日志和最终分数，便于与论文表格对齐。

---

## Acknowledgement

This code refers to [G-designer](https://github.com/yanweiyue/GDesigner).

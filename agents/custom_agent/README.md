# 自定义智能体目录

## 用途

本目录提供可独立运行的 **`CustomAgent`**（规划 → 执行 → 反思）。既可单独拷贝开源/嵌入你的项目，也可放在完整 **`custom_agent_test_package`** 中与评测脚本、数据集一起使用。

## 目录结构

```
agents/custom_agent/
├── README.md           # 本文件
├── requirements.txt    # 仅跑本 Agent 的最小依赖
├── .gitignore
├── __init__.py
├── agent.py            # 智能体实现（核心）
├── config.yaml         # 说明用配置（与代码默认值对照；当前实现以代码与环境变量为准）
└── tools/              # 自定义扩展示例（可选；主逻辑工具在 agent.py 内）
    ├── __init__.py
    └── my_tool.py
```

## 快速开始（只带本目录时）

1. 安装依赖：

```bash
pip install -r requirements.txt
```

2. 配置通义千问兼容接口（默认）：

```bash
export DASHSCOPE_API_KEY="sk-你的Key"
```

3. 在本目录下自检：

```bash
python agent.py
```

也可在其他项目中把本目录加入 `sys.path` 后：

```python
from agent import CustomAgent

agent = CustomAgent()
print(agent.run("计算 3+5"))
```

若希望包名为 `agents.custom_agent`（与完整测试包一致），请保留上级目录 **`agents/custom_agent/`** 这一层级，并在工程根执行 `python` 或将工程根加入 `PYTHONPATH`。

## 放在完整测试包内时

仓库根目录另有 **`benchmarks/`、`evaluators/`、`scripts/`** 等。评测类脚本会从包根导入，例如：

```python
from agents.custom_agent.agent import CustomAgent
```

一键流程见仓库根 **`README.md`** 与 **`run_all_tests.sh`**。本仓库**没有** `agents/agent_factory.py`；若你在自己的系统里用工厂模式注册 Agent，自行把 `CustomAgent` 映射到你的注册表即可。

## 配置说明

- **API Key**：构造参数 `api_key` 优先，否则读环境变量 **`DASHSCOPE_API_KEY`**。
- **模型与端点**：构造参数或环境变量 **`CUSTOM_AGENT_LLM_MODEL`**、**`CUSTOM_AGENT_LLM_API_BASE`** 等（见 `agent.py` 中 `_env_str` / `CustomAgent.__init__`）。
- **`config.yaml`**：便于文档与运维对齐默认模型、`api_key_env` 等；**运行时以代码与环境变量为准**，与 yaml 不一致时以代码为准。

## 标准接口说明

| 方法 | 说明 |
|------|------|
| `__init__(api_key=None, model=None, **kwargs)` | 初始化 |
| `run(task: str, context: list \| None) -> dict` | 执行任务，必填 |
| `reset()` | 清空对话历史等状态 |

`run()` 至少包含：`success`、`output`、`error`。部分路径会附带 **`_meta`**（子任务统计、LLM 调用次数等），接入评测时可按需读取。

## 开源发布前建议

- 勿提交 **`__pycache__/`**、**`*.pyc`**、本地 **`.env`**（本目录 `.gitignore` 已忽略常见项）。
- 在仓库根或本目录补充 **许可证（LICENSE）** 说明。
- 勿将真实 **API Key** 写入仓库。

## 示例实现

本仓库中的 **`CustomAgent`** 即生产级示例；修改行为请优先编辑 **`agent.py`**，扩展工具可放在 **`tools/`** 并在 `agent.py` 中接入。

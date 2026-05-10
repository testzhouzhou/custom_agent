#!/usr/bin/env python3
"""
规划-执行-反思 智能体

架构：
  输入任务
    ↓
  [规划] LLM 分解为子任务，建立依赖图
    ↓
  [执行] 按依赖顺序执行子任务（可并行的并行执行）
    ↓
  [反思] 检查结果
    ├─ 成功 → 继续下一个
    ├─ 失败 → 重试（最多3次）或重新规划
    └─ 部分成功 → 调整后续计划
    ↓
  [总结] 汇总所有子任务结果

内置工具（与 ToolRegistry.TOOLS 一致）：
  - search: 知识库/业务检索（内置电商 Mock：销售维度查询、生成类报告请求会返回结构化摘要/Markdown）
  - calculator: 数学计算
  - code_executor: Python 代码执行
  - memory_store: 记忆存储/读取
  - web_fetch: 网页内容获取
  - safety_checker: 安全检测（有害内容/Prompt注入/Jailbreak/隐私）
"""

import os
import re
import json
import math
import builtins
import ast
import time
import subprocess
import operator
import requests
from typing import List, Dict, Optional, Any, Union
from dataclasses import dataclass, field
from urllib.parse import urlparse


# ---------- 可调限制与配置键（环境变量前缀 CUSTOM_AGENT_）----------
_MAX_TASK_LEN = int(os.getenv("CUSTOM_AGENT_MAX_TASK_LEN", "100000"))
_MAX_TOOL_INPUT_LEN = int(os.getenv("CUSTOM_AGENT_MAX_TOOL_INPUT_LEN", "20000"))
_MAX_CODE_LEN = int(os.getenv("CUSTOM_AGENT_MAX_CODE_LEN", "50000"))
_MAX_FETCH_URL_LEN = int(os.getenv("CUSTOM_AGENT_MAX_FETCH_URL_LEN", "2048"))
_MEMORY_KEY_RE = re.compile(r"^[A-Za-z0-9_.-]{1,256}$")

_ENV_PREFIX = "CUSTOM_AGENT_"


def _env_str(name: str, default: str) -> str:
    key = f"{_ENV_PREFIX}{name}"
    v = os.getenv(key) or os.getenv(name)
    return default if v is None or str(v).strip() == "" else str(v).strip()


def _env_int(name: str, default: int) -> int:
    key = f"{_ENV_PREFIX}{name}"
    raw = os.getenv(key) or os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    key = f"{_ENV_PREFIX}{name}"
    raw = os.getenv(key) or os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _http_error_detail(response: requests.Response) -> str:
    parts = [f"HTTP {response.status_code}"]
    if response.reason:
        parts.append(response.reason)
    try:
        body = response.json()
        if isinstance(body, dict):
            err = body.get("error")
            if err is not None:
                parts.append(json.dumps(err, ensure_ascii=False) if not isinstance(err, str) else err)
            elif body.get("message"):
                parts.append(str(body["message"]))
    except (ValueError, json.JSONDecodeError):
        text = (response.text or "")[:800].strip()
        if text:
            parts.append(text)
    return " | ".join(p for p in parts if p)


def _validate_http_url(url: str) -> tuple:
    """校验 web_fetch 使用的 URL。"""
    try:
        parsed = urlparse(url)
    except ValueError:
        return False, "URL 解析失败"
    if parsed.scheme not in ("http", "https"):
        return False, "仅允许 http 或 https 协议"
    if not parsed.netloc:
        return False, "URL 缺少有效主机名"
    host = (parsed.hostname or "").lower()
    if host in ("localhost", "127.0.0.1", "::1", "0.0.0.0"):
        return False, "不允许请求本机回环地址"
    return True, ""


# ---------- 安全数学表达式求值（AST，禁止 eval）----------
_ALLOWED_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod, ast.FloorDiv)
_ALLOWED_UNARY = (ast.UAdd, ast.USub)
_BUILTIN_CALLABLES = frozenset({"abs", "round", "min", "max", "sum", "pow"})


def _safe_eval_math_ast(node: ast.AST) -> Union[int, float]:
    """仅允许数值运算、math 模块与少量内置函数。"""
    if isinstance(node, ast.Expression):
        return _safe_eval_math_ast(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool):
            raise ValueError("不允许布尔字面量")
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("仅支持 int/float 常量")
    # Python 3.8 兼容：旧版 Num（本包要求 3.11+，保留无妨）
    if isinstance(node, ast.Num):  # pragma: no cover
        return node.n
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, _ALLOWED_UNARY):
        v = _safe_eval_math_ast(node.operand)
        return +v if isinstance(node.op, ast.UAdd) else -v
    if isinstance(node, ast.BinOp) and isinstance(node.op, _ALLOWED_BINOPS):
        left = _safe_eval_math_ast(node.left)
        right = _safe_eval_math_ast(node.right)
        op_type = type(node.op)
        if op_type is ast.Add:
            return operator.add(left, right)
        if op_type is ast.Sub:
            return operator.sub(left, right)
        if op_type is ast.Mult:
            return operator.mul(left, right)
        if op_type is ast.Div:
            return operator.truediv(left, right)
        if op_type is ast.Pow:
            return operator.pow(left, right)
        if op_type is ast.Mod:
            return operator.mod(left, right)
        if op_type is ast.FloorDiv:
            return operator.floordiv(left, right)
    if isinstance(node, ast.Call):
        return _safe_eval_math_call(node)
    raise ValueError(f"不允许的表达式语法: {type(node).__name__}")


def _safe_eval_math_call(node: ast.Call) -> Union[int, float]:
    if node.keywords:
        raise ValueError("不允许关键字参数")
    func = node.func

    if isinstance(func, ast.Name) and func.id in _BUILTIN_CALLABLES:
        bid = func.id
        if bid == "sum":
            if len(node.args) != 1:
                raise ValueError("sum 只接受一个列表或元组参数")
            arg0 = node.args[0]
            if not isinstance(arg0, (ast.List, ast.Tuple)):
                raise ValueError("sum 只接受显式列表或元组字面量")
            seq = [_safe_eval_math_ast(elt) for elt in arg0.elts]
            if any(isinstance(x, float) for x in seq):
                return float(sum(seq))
            return sum(seq)
        if bid in ("min", "max"):
            fn = getattr(builtins, bid)
            if len(node.args) == 1 and isinstance(node.args[0], (ast.List, ast.Tuple)):
                seq = [_safe_eval_math_ast(elt) for elt in node.args[0].elts]
                if not seq:
                    raise ValueError(f"{bid} 参数不能为空序列")
                return fn(seq)
            args = [_safe_eval_math_ast(a) for a in node.args]
            if not args:
                raise ValueError(f"{bid} 至少需要一个参数")
            return fn(*args)
        args = [_safe_eval_math_ast(a) for a in node.args]
        fn = getattr(builtins, bid, None)
        if fn is None or not callable(fn):
            raise ValueError(f"未授权函数: {bid}")
        return fn(*args)

    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) and func.value.id == "math":
        name = func.attr
        if name.startswith("_"):
            raise ValueError("不允许访问 math 私有成员")
        attr = getattr(math, name, None)
        if attr is None:
            raise ValueError(f"math 无此成员: {name}")
        args = [_safe_eval_math_ast(a) for a in node.args]
        if callable(attr):
            return attr(*args)
        if len(args) == 0 and isinstance(attr, (int, float)):
            return attr
        raise ValueError(f"math.{name} 调用参数不合法")

    raise ValueError("仅允许 abs/round/min/max/sum/pow 或 math.* 调用")


def safe_eval_math_expression(expression: str) -> Union[int, float]:
    expr = expression.strip()
    if not expr:
        raise ValueError("表达式为空")
    if len(expr) > 4096:
        raise ValueError("表达式过长")
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"表达式语法错误: {e}") from e
    return _safe_eval_math_ast(tree)


# ============================================================
#  业务背景
# ============================================================

BUSINESS_CONTEXT = """
你是一个电商数据分析智能体，服务于一家中型电商平台。
你的职责是：分析销售数据、识别异常趋势、生成业务报告、回答数据相关问题。

业务知识：
- 核心指标：GMV（成交总额）、客单价、转化率、复购率、退货率
- 数据维度：品类、地区、时间（日/周/月/季度）、渠道（APP/小程序/PC）
- 常见分析场景：月度销售报告、促销效果评估、异常预警、品类对比
- 安全要求：客户数据（手机号、身份证号、收货地址）必须脱敏，不得在报告中明文展示
"""


# ============================================================
#  数据类
# ============================================================

@dataclass
class SubTask:
    """子任务"""
    id: str
    description: str
    depends_on: List[str] = field(default_factory=list)
    tool: str = ""
    tool_input: str = ""
    result: str = ""
    status: str = "pending"  # pending / running / success / failed / skipped
    retry_count: int = 0
    max_retries: int = 3


@dataclass
class Plan:
    """任务规划"""
    original_task: str = ""
    subtasks: List[SubTask] = field(default_factory=list)
    execution_order: List[str] = field(default_factory=list)
    parallel_groups: List[List[str]] = field(default_factory=list)


@dataclass
class AgentState:
    """智能体运行状态"""
    original_task: str = ""
    plan: Plan = None
    current_phase: str = ""  # planning / executing / reflecting / summarizing
    subtask_results: Dict[str, str] = field(default_factory=dict)
    memory: Dict[str, str] = field(default_factory=dict)
    final_answer: str = ""
    total_tokens: int = 0
    total_llm_calls: int = 0
    start_time: float = 0.0
    reflection_log: List[str] = field(default_factory=list)
    replan_count: int = 0


# ============================================================
#  内置工具
# ============================================================

class ToolRegistry:
    """工具注册中心"""

    TOOLS = {
        "search": {
            "name": "search",
            "description": (
                "搜索知识库或通用信息。内置评测用 Mock："
                "① 销售/品类/地区/渠道/订单等经营类查询 → 结构化摘要；"
                "② 含「生成/撰写/输出」且含「报告/周报/月报」→ Markdown 报告模板。"
                "其他查询返回通用占位说明。输入：自然语言"
            ),
            "usage": "search: 查询2024年6月各品类销售额  或  search: 生成月度销售报告, 含GMV与异常预警",
        },
        "calculator": {
            "name": "calculator",
            "description": "执行数学计算。输入：数学表达式，如 2+3*4",
            "usage": "calculator: 2 + 3 * 4",
        },
        "code_executor": {
            "name": "code_executor",
            "description": "执行 Python 代码。输入：Python 代码片段",
            "usage": 'code_executor: print("hello")',
        },
        "memory_store": {
            "name": "memory_store",
            "description": "存储或读取记忆。输入：key=value 存储，key 读取",
            "usage": "memory_store: mykey=hello 或 memory_store: mykey",
        },
        "web_fetch": {
            "name": "web_fetch",
            "description": "获取网页文本内容。输入：URL",
            "usage": "web_fetch: https://example.com",
        },
        "safety_checker": {
            "name": "safety_checker",
            "description": "检测输入是否包含有害内容、Prompt注入或 Jailbreak 尝试。输入：待检测文本",
            "usage": "safety_checker: 待检测文本",
        },
    }

    @classmethod
    def get_tool_descriptions(cls) -> str:
        lines = []
        for name, info in cls.TOOLS.items():
            lines.append(f"  - {info['name']}: {info['description']}")
            lines.append(f"    用法: {info['usage']}")
        return "\n".join(lines)

    @classmethod
    def execute(cls, tool_name: str, tool_input: str, state: AgentState) -> str:
        tool_name = (tool_name or "").strip().lower()
        tool_input = (tool_input or "").strip()
        if len(tool_input) > _MAX_TOOL_INPUT_LEN:
            return f"错误: 工具输入过长（上限 {_MAX_TOOL_INPUT_LEN} 字符）"

        try:
            if tool_name == "search":
                return cls._search(tool_input)
            elif tool_name == "calculator":
                return cls._calculator(tool_input)
            elif tool_name == "code_executor":
                return cls._code_executor(tool_input)
            elif tool_name == "memory_store":
                return cls._memory_store(tool_input, state)
            elif tool_name == "web_fetch":
                return cls._web_fetch(tool_input)
            elif tool_name == "safety_checker":
                return cls._safety_checker(tool_input)
            else:
                return f"错误: 未知工具 '{tool_name}'。可用工具: {', '.join(cls.TOOLS.keys())}"
        except Exception as e:
            return f"错误: 工具执行失败 — {e}"

    @staticmethod
    def _search(query: str) -> str:
        """知识库检索：通用查询 + 内置电商销售/报告 Mock（统一走 search，避免额外工具名）。"""
        q = (query or "").strip()
        if len(q) > _MAX_TOOL_INPUT_LEN:
            return f"错误: 搜索关键词过长（上限 {_MAX_TOOL_INPUT_LEN} 字符）"
        ql = q.lower()
        # 报告类（原 report_generator）
        if ("报告" in q or "周报" in q or "月报" in q or "markdown" in ql) and (
            "生成" in q or "撰写" in q or "输出" in q
        ):
            return ToolRegistry._mock_markdown_report(q)
        # 销售/经营数据类（原 sales_db_query）
        if ToolRegistry._query_looks_like_sales(q, ql):
            return ToolRegistry._mock_sales_kb(q)
        return f"[搜索结果] 已收到查询: '{q}'。请基于你的知识继续分析。"

    @staticmethod
    def _query_looks_like_sales(q: str, ql: str) -> bool:
        cn = (
            "销售", "品类", "地区", "渠道", "订单", "客单价", "退货", "华东", "华南", "华北",
            "西南", "月度", "趋势", "转化", "查询", "同比", "环比", "销售额",
        )
        en = ("category", "region", "channel", "sales", "select", " from ", "month", "gmv", "order")
        return any(x in q for x in cn) or any(x in ql for x in en)

    @staticmethod
    def _mock_markdown_report(raw: str) -> str:
        """模拟生成结构化业务报告（Markdown）。"""
        if len(raw) > _MAX_TOOL_INPUT_LEN:
            return f"错误: 报告参数过长（上限 {_MAX_TOOL_INPUT_LEN} 字符）"
        lines = raw.split(",", 1)
        title = lines[0].strip() if lines else "业务报告"
        report = f"""# {title}

## 数据概览
- 统计周期：2024年1月 - 2024年6月
- 总销售额：4,710,000 元
- 总订单量：42,100 单
- 平均客单价：111.9 元

## 核心指标
| 指标 | 数值 | 环比变化 |
|------|------|---------|
| GMV | 4,710,000 元 | +12.3% |
| 客单价 | 111.9 元 | +5.2% |
| 转化率 | 2.7% | -0.3% |
| 退货率 | 5.7% | +0.8% |

## 异常预警
- 服装鞋帽品类退货率 12.8%，高于行业平均水平（8%）
- PC 渠道转化率持续下降（1.5% → 1.2%），建议优化 PC 端体验

## 建议
1. 重点优化服装鞋帽品类的退换货流程
2. 加大 APP 渠道投入（转化率最高，占比持续提升）
3. 关注华南地区销售增长放缓趋势

> 本报告由电商数据分析智能体自动生成
"""
        return f"[报告生成] 已生成报告: {title}\n{report}"

    @staticmethod
    def _mock_sales_kb(query: str) -> str:
        """
        模拟电商销售知识库命中（结构化摘要，可评测用）
        """
        q = (query or "").strip()
        if len(q) > _MAX_TOOL_INPUT_LEN:
            return f"错误: 查询内容过长（上限 {_MAX_TOOL_INPUT_LEN} 字符）"
        query_lower = q.lower()

        mock_data = {
            "品类": {
                "电子产品": {"销售额": 1250000, "订单量": 3200, "客单价": 390.6, "退货率": 5.2},
                "服装鞋帽": {"销售额": 890000, "订单量": 8900, "客单价": 100.0, "退货率": 12.8},
                "食品饮料": {"销售额": 560000, "订单量": 15600, "客单价": 35.9, "退货率": 2.1},
                "家居日用": {"销售额": 720000, "订单量": 5400, "客单价": 133.3, "退货率": 4.5},
                "美妆护肤": {"销售额": 430000, "订单量": 6200, "客单价": 69.4, "退货率": 3.8},
            },
            "地区": {
                "华东": {"销售额": 1680000, "订单量": 14200, "占比": 35.2},
                "华南": {"销售额": 980000, "订单量": 8500, "占比": 20.5},
                "华北": {"销售额": 850000, "订单量": 7200, "占比": 17.8},
                "西南": {"销售额": 520000, "订单量": 4800, "占比": 10.9},
                "其他": {"销售额": 730000, "订单量": 7400, "占比": 15.3},
            },
            "趋势": {
                "1月": 920000, "2月": 780000, "3月": 1050000,
                "4月": 1120000, "5月": 1280000, "6月": 1560000,
            },
            "渠道": {
                "APP": {"销售额": 2350000, "占比": 49.2, "转化率": 3.8},
                "小程序": {"销售额": 1480000, "占比": 31.0, "转化率": 2.9},
                "PC": {"销售额": 930000, "占比": 19.5, "转化率": 1.5},
            },
        }

        result_lines = [f"[销售数据库] 查询: {q}"]

        if "品类" in query_lower or "category" in query_lower:
            result_lines.append("\n各品类销售数据:")
            for name, data in mock_data["品类"].items():
                result_lines.append(
                    f"  {name}: 销售额 {data['销售额']:,}元 | "
                    f"订单量 {data['订单量']:,} | "
                    f"客单价 {data['客单价']:.1f}元 | "
                    f"退货率 {data['退货率']}%"
                )
        elif "地区" in query_lower or "region" in query_lower:
            result_lines.append("\n各地区销售数据:")
            for name, data in mock_data["地区"].items():
                result_lines.append(
                    f"  {name}: 销售额 {data['销售额']:,}元 | "
                    f"订单量 {data['订单量']:,} | "
                    f"占比 {data['占比']}%"
                )
        elif "趋势" in query_lower or "月度" in query_lower or "month" in query_lower:
            result_lines.append("\n月度销售趋势:")
            for month, amount in mock_data["趋势"].items():
                result_lines.append(f"  {month}: {amount:,}元")
        elif "渠道" in query_lower or "channel" in query_lower:
            result_lines.append("\n各渠道销售数据:")
            for name, data in mock_data["渠道"].items():
                result_lines.append(
                    f"  {name}: 销售额 {data['销售额']:,}元 | "
                    f"占比 {data['占比']}% | "
                    f"转化率 {data['转化率']}%"
                )
        else:
            result_lines.append("\n全量数据概览:")
            total_sales = sum(d["销售额"] for d in mock_data["品类"].values())
            total_orders = sum(d["订单量"] for d in mock_data["品类"].values())
            result_lines.append(f"  总销售额: {total_sales:,}元")
            result_lines.append(f"  总订单量: {total_orders:,}")
            result_lines.append(f"  平均客单价: {total_sales / total_orders:.1f}元")

        return "\n".join(result_lines)

    @staticmethod
    def _calculator(expression: str) -> str:
        expr = (expression or "").strip()
        if not expr:
            return "错误: 表达式为空"
        if len(expr) > 4096:
            return "错误: 表达式过长（上限 4096 字符）"
        if not re.match(r"^[\d\s\+\-\*\/\(\)\.\%\*\*a-zA-Z_,]+$", expr):
            return f"错误: 表达式包含不允许的字符: {expr!r}"
        try:
            result = safe_eval_math_expression(expr)
            return f"[计算结果] {expr} = {result}"
        except Exception as e:
            return f"错误: 计算失败 — {e}"

    @staticmethod
    def _code_executor(code: str) -> str:
        if len(code) > _MAX_CODE_LEN:
            return f"错误: 代码过长（上限 {_MAX_CODE_LEN} 字符）"
        dangerous = ["import os", "import sys", "import subprocess",
                     "eval(", "exec(", "__import__", "open(", "file("]
        for word in dangerous:
            if word in code:
                return f"错误: 代码包含禁止的操作 '{word}'"
        try:
            result = subprocess.run(
                ["python3", "-c", code],
                capture_output=True, text=True, timeout=10,
                env={**os.environ, "PYTHONPATH": ""},
            )
            if result.returncode == 0:
                output = result.stdout.strip() or "(无输出)"
                return f"[代码执行结果]\n{output}"
            else:
                return f"错误: 代码执行失败\n{result.stderr.strip()}"
        except subprocess.TimeoutExpired:
            return "错误: 代码执行超时（10秒限制）"
        except Exception as e:
            return f"错误: {e}"

    @staticmethod
    def _memory_store(input_str: str, state: AgentState) -> str:
        if "=" in input_str:
            key, value = input_str.split("=", 1)
            key = key.strip()
            value = value.strip()
            if not _MEMORY_KEY_RE.match(key):
                return "错误: 记忆键仅允许字母、数字、下划线、点、连字符，长度 1-256"
            if len(value) > _MAX_TOOL_INPUT_LEN:
                return f"错误: 记忆值过长（上限 {_MAX_TOOL_INPUT_LEN} 字符）"
            state.memory[key] = value
            return f"[记忆存储] 已保存: {key} = {value}"
        else:
            key = input_str.strip()
            if not _MEMORY_KEY_RE.match(key):
                return "错误: 记忆键仅允许字母、数字、下划线、点、连字符，长度 1-256"
            if key in state.memory:
                return f"[记忆读取] {key} = {state.memory[key]}"
            else:
                return f"[记忆读取] 未找到: {key}"

    @staticmethod
    def _web_fetch(url: str) -> str:
        raw = (url or "").strip()
        if not raw:
            return "错误: URL 为空"
        if len(raw) > _MAX_FETCH_URL_LEN:
            return f"错误: URL 过长（上限 {_MAX_FETCH_URL_LEN} 字符）"
        ok, reason = _validate_http_url(raw)
        if not ok:
            return f"错误: {reason}"
        try:
            resp = requests.get(
                raw,
                timeout=10,
                headers={"User-Agent": "Mozilla/5.0 (compatible; Agent/1.0)"},
            )
            if resp.status_code != 200:
                detail = _http_error_detail(resp)
                return f"错误: 网页请求失败 — {detail}"
            text = re.sub(r"<[^>]+>", "", resp.text)
            text = re.sub(r"\s+", " ", text).strip()
            if len(text) > 2000:
                text = text[:2000] + "...(已截断)"
            return f"[网页内容] {raw}\n{text}"
        except requests.Timeout:
            return "错误: 网页获取超时（10 秒）"
        except requests.RequestException as e:
            return f"错误: 网络请求异常 — {type(e).__name__}: {e}"
        except Exception as e:
            return f"错误: 网页获取失败 — {type(e).__name__}: {e}"

    @staticmethod
    def _safety_checker(text: str) -> str:
        """
        安全检测 — 基于规则的输入过滤

        检测 4 类攻击向量：
        1. 有害内容（暴力、色情、仇恨）
        2. Prompt 注入（覆盖系统提示）
        3. Jailbreak（角色扮演绕过安全限制）
        4. 隐私泄露（身份证号、手机号等）
        """
        t = (text or "").strip()
        if len(t) > _MAX_TOOL_INPUT_LEN:
            return f"错误: 待检测文本过长（上限 {_MAX_TOOL_INPUT_LEN} 字符）"
        text = t
        threats = []

        # 1. 有害内容关键词
        harmful_patterns = [
            r"(暴力|恐怖|炸弹|毒药|自杀|自残|杀人|强奸|虐杀)",
            r"(色情|裸体|性交|淫秽|色情网站)",
            r"(种族歧视|性别歧视|仇恨言论|歧视性)",
        ]
        for pattern in harmful_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                threats.append("有害内容")
                break

        # 2. Prompt 注入检测
        injection_patterns = [
            r"(忽略.*指令|忽略.*提示|忽略.*规则)",
            r"(忽略.*之前|forget.*previous|ignore.*above)",
            r"(你现在是|你现在是.*没有.*限制)",
            r"(system.*prompt|系统提示|system instruction)",
            r"(绕过.*安全|bypass.*security|jailbreak)",
            r"(DAN|Do Anything Now|Developer Mode)",
        ]
        for pattern in injection_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                threats.append("Prompt 注入")
                break

        # 3. Jailbreak 检测
        jailbreak_patterns = [
            r"(假设.*没有.*限制|假设.*不受.*约束)",
            r"(角色扮演.*安全|role.*play.*security)",
            r"(道德.*约束|ethical.*constraint|道德约束)",
            r"(编码.*绕过|encode.*bypass|base64.*decode)",
            r"(转义.*检测|escape.*detection|e\s*s\s*c\s*a\s*p\s*e)",
        ]
        for pattern in jailbreak_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                threats.append("Jailbreak 尝试")
                break

        # 4. 隐私信息检测
        privacy_patterns = [
            r"(\d{17}[\dXx]|\d{15})",  # 身份证号
            r"(1[3-9]\d{9})",  # 手机号
            r"(\d{6}@\d{4})",  # 学号/工号
        ]
        for pattern in privacy_patterns:
            if re.search(pattern, text):
                threats.append("隐私信息")
                break

        if threats:
            threat_str = "、".join(set(threats))
            return f"[安全检测] 检测到威胁: {threat_str}。建议拒绝执行。"
        else:
            return "[安全检测] 输入安全，未发现威胁。"


# ============================================================
#  规划-执行-反思 智能体
# ============================================================

class CustomAgent:
    """
    规划-执行-反思 智能体

    标准接口：
      - __init__(api_key, model, **kwargs)
      - run(task, context) -> {"success": bool, "output": str, "error": str}
      - reset()

    配置可通过构造参数或环境变量覆盖（前缀 CUSTOM_AGENT_）：
      LLM_MODEL, LLM_API_BASE, LLM_TEMPERATURE, LLM_TIMEOUT, LLM_MAX_TOKENS
    """

    # ---- Prompt: 规划阶段 ----
    PLANNING_PROMPT = """你是一个电商数据分析智能体。用户给你一个任务，你需要：

1. 将任务分解为多个子任务
2. 为每个子任务选择合适的工具
3. 建立子任务之间的依赖关系

业务背景：
{business_context}

可用工具：
{tools}

输出格式（JSON）：
```json
[
  {{
    "id": "task_1",
    "description": "子任务描述",
    "depends_on": [],
    "tool": "工具名",
    "tool_input": "工具参数"
  }},
  {{
    "id": "task_2",
    "description": "子任务描述",
    "depends_on": ["task_1"],
    "tool": "工具名",
    "tool_input": "工具参数"
  }}
]
```

规则：
- 子任务 ID 从 task_1 开始编号
- depends_on 是依赖的子任务 ID 列表，无依赖填 []
- 如果某个子任务不需要工具就能完成，tool 填 "none"，tool_input 填 "直接回答"
- 尽量让不相互依赖的子任务并行执行
- 子任务数量控制在 3-8 个
- 涉及客户数据时，注意脱敏处理

当前任务：
{task}

对话历史：
{context}

请输出规划："""

    # ---- Prompt: 反思阶段 ----
    REFLECTION_PROMPT = """你是一个质量检查专家。一个智能体正在执行任务，已完成了一些子任务。

原始任务：
{task}

已完成子任务结果：
{results}

请检查：
1. 已完成的子任务结果是否合理
2. 是否需要调整后续计划（如增加子任务、修改工具选择）
3. 当前进度是否足够生成最终答案

输出格式（JSON）：
\`\`\`json
{{
  "status": "continue" | "replan" | "done",
  "reason": "判断理由",
  "adjustments": "如果需要调整，说明调整内容；否则为空"
}}
\`\`\`

请输出反思："""

    # ---- Prompt: 总结阶段 ----
    SUMMARY_PROMPT = """你是一个任务总结专家。一个智能体已完成所有子任务，请汇总结果并给出最终答案。

原始任务：
{task}

子任务结果：
{results}

请给出清晰、完整的最终答案："""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None, **kwargs):
        # API Key：显式参数 > 环境变量 DASHSCOPE_API_KEY
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        # 模型与端点：显式参数 > CUSTOM_AGENT_LLM_* 环境变量 > 默认值
        self.model = model if model is not None else _env_str("LLM_MODEL", "qwen-plus")
        self.api_base = (
            kwargs.get("api_base")
            or _env_str("LLM_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        ).rstrip("/")
        self.max_retries = kwargs.get("max_retries", 3)
        self.max_replans = kwargs.get("max_replans", 2)
        if kwargs.get("temperature") is not None:
            self.temperature = float(kwargs["temperature"])
        else:
            self.temperature = _env_float("LLM_TEMPERATURE", 0.7)
        self.llm_timeout = int(kwargs.get("llm_timeout") or _env_int("LLM_TIMEOUT", 60))
        self.llm_max_tokens = int(kwargs.get("llm_max_tokens") or _env_int("LLM_MAX_TOKENS", 2048))
        self.enable_tools = kwargs.get("enable_tools", True)
        # 上下文窗口管理
        self.max_context_turns = kwargs.get("max_context_turns", 10)
        self.enable_safety_check = kwargs.get("enable_safety_check", True)
        # 业务背景（默认使用电商数据分析）
        self.business_context = kwargs.get("business_context", BUSINESS_CONTEXT)
        self._context_history: List[Dict] = []  # 持久化对话历史
        self._last_llm_error: Optional[str] = None

    def run(self, task: str, context: List[Dict] = None) -> Dict:
        """
        执行任务 — 规划 → 执行 → 反思 → 总结

        Args:
            task: 任务描述
            context: 对话历史（可选，与内部历史合并）

        Returns:
            {"success": True, "output": "...", "error": None}
        """
        task_err = self._validate_task_input(task)
        if task_err:
            return {"success": False, "output": "", "error": task_err}

        ctx_err = self._validate_context(context)
        if ctx_err:
            return {"success": False, "output": "", "error": ctx_err}

        self._last_llm_error = None

        # 安全检测（在规划之前）
        if self.enable_safety_check:
            safety_result = self._check_safety(task)
            if safety_result["blocked"]:
                return {
                    "success": False,
                    "output": "",
                    "error": f"安全拦截: {safety_result['threats']}",
                    "_meta": {"safety_check": safety_result},
                }

        # 合并上下文
        if context:
            self._context_history.extend(context)
        # 截断到最大轮数
        if len(self._context_history) > self.max_context_turns:
            self._context_history = self._context_history[-self.max_context_turns:]

        state = AgentState(
            original_task=task,
            start_time=time.time(),
        )

        try:
            # ========== 阶段 1: 规划 ==========
            state.current_phase = "planning"
            plan = self._plan(task, state, context)
            if not plan:
                detail = self._last_llm_error or "LLM 无可用响应"
                return {"success": False, "output": "", "error": f"规划失败：{detail}"}
            state.plan = plan

            # ========== 阶段 2: 执行 + 反思 循环 ==========
            for replan_round in range(self.max_replans + 1):
                state.replan_count = replan_round

                # 执行子任务
                state.current_phase = "executing"
                self._execute_plan(plan, state)

                # 反思
                state.current_phase = "reflecting"
                reflection = self._reflect(task, plan, state)

                if reflection["status"] == "done":
                    break
                elif reflection["status"] == "replan" and replan_round < self.max_replans:
                    # 重新规划
                    state.reflection_log.append(f"第 {replan_round + 1} 次重新规划: {reflection.get('adjustments', '')}")
                    plan = self._plan_with_context(task, plan, reflection, state, context)
                    if not plan:
                        break
                # else: continue 执行剩余子任务

            # ========== 阶段 3: 总结 ==========
            state.current_phase = "summarizing"
            final_answer = self._summarize(task, plan, state)
            state.final_answer = final_answer

            # ========== 返回结果 ==========
            elapsed = time.time() - state.start_time
            summary = self._build_summary(state, elapsed)

            return {
                "success": True,
                "output": final_answer or summary,
                "error": None,
                "_meta": {
                    "tokens": state.total_tokens,
                    "llm_calls": state.total_llm_calls,
                    "elapsed": round(elapsed, 2),
                    "replans": state.replan_count,
                    "subtasks_total": len(plan.subtasks),
                    "subtasks_success": sum(1 for s in plan.subtasks if s.status == "success"),
                    "subtasks_failed": sum(1 for s in plan.subtasks if s.status == "failed"),
                    "memory": state.memory,
                    "reflection_log": state.reflection_log,
                    "context_turns": len(self._context_history),
                    "max_context_turns": self.max_context_turns,
                    "safety_checked": self.enable_safety_check,
                    "subtasks": [
                        {
                            "id": s.id,
                            "description": s.description,
                            "tool": s.tool,
                            "status": s.status,
                            "retry_count": s.retry_count,
                            "result": s.result[:200] if s.result else "",
                        }
                        for s in plan.subtasks
                    ],
                },
            }

        except Exception as e:
            return {"success": False, "output": "", "error": str(e)}

    def reset(self):
        """重置智能体状态"""
        self._context_history = []
        if hasattr(self, 'state'):
            self.state = None

    # ============================================================
    #  安全检测
    # ============================================================

    def _validate_task_input(self, task: Any) -> Optional[str]:
        if task is None:
            return "任务描述不能为空"
        if not isinstance(task, str):
            return "任务描述必须是字符串"
        t = task.strip()
        if not t:
            return "任务描述不能为空"
        if len(t) > _MAX_TASK_LEN:
            return f"任务描述过长（上限 {_MAX_TASK_LEN} 字符）"
        return None

    def _validate_context(self, context: Any) -> Optional[str]:
        if context is None:
            return None
        if not isinstance(context, list):
            return "参数 context 必须是消息字典的列表"
        for i, msg in enumerate(context):
            if not isinstance(msg, dict):
                return f"context[{i}] 必须是字典"
            role = msg.get("role", "user")
            if not isinstance(role, str):
                return f"context[{i}].role 必须是字符串"
            content = msg.get("content", "")
            if content is not None and not isinstance(content, str):
                return f"context[{i}].content 必须是字符串"
            if isinstance(content, str) and len(content) > _MAX_TOOL_INPUT_LEN:
                return f"context[{i}] 内容过长（上限 {_MAX_TOOL_INPUT_LEN} 字符）"
        return None

    def _check_safety(self, text: str) -> Dict:
        """
        输入安全检测 — 在规划前执行

        Returns:
            {"blocked": bool, "threats": List[str], "details": str}
        """
        if not isinstance(text, str):
            return {"blocked": True, "threats": ["非法输入类型"], "details": "任务必须是字符串"}
        if len(text) > _MAX_TASK_LEN:
            return {"blocked": True, "threats": ["输入过长"], "details": f"超过 {_MAX_TASK_LEN} 字符"}

        threats = []

        # 1. 有害内容关键词
        harmful_patterns = [
            r"(暴力|恐怖|炸弹|毒药|自杀|自残|杀人|强奸|虐杀)",
            r"(色情|裸体|性交|淫秽|色情网站)",
            r"(种族歧视|性别歧视|仇恨言论|歧视性)",
        ]
        for pattern in harmful_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                threats.append("有害内容")
                break

        # 2. Prompt 注入检测
        injection_patterns = [
            r"(忽略.*指令|忽略.*提示|忽略.*规则)",
            r"(忽略.*之前|forget.*previous|ignore.*above)",
            r"(你现在是|你现在是.*没有.*限制)",
            r"(system.*prompt|系统提示|system instruction)",
            r"(绕过.*安全|bypass.*security|jailbreak)",
            r"(DAN|Do Anything Now|Developer Mode)",
        ]
        for pattern in injection_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                threats.append("Prompt 注入")
                break

        # 3. Jailbreak 检测
        jailbreak_patterns = [
            r"(假设.*没有.*限制|假设.*不受.*约束)",
            r"(角色扮演.*安全|role.*play.*security)",
            r"(道德.*约束|ethical.*constraint|道德约束)",
            r"(编码.*绕过|encode.*bypass|base64.*decode)",
            r"(转义.*检测|escape.*detection)",
        ]
        for pattern in jailbreak_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                threats.append("Jailbreak 尝试")
                break

        # 4. 隐私信息检测
        privacy_patterns = [
            r"(\d{17}[\dXx]|\d{15})",  # 身份证号
            r"(1[3-9]\d{9})",  # 手机号
        ]
        for pattern in privacy_patterns:
            if re.search(pattern, text):
                threats.append("隐私信息")
                break

        blocked = len(threats) > 0
        return {
            "blocked": blocked,
            "threats": threats,
            "details": f"检测到: {', '.join(set(threats))}" if threats else "输入安全",
        }

    # ============================================================
    #  上下文窗口管理
    # ============================================================

    def _build_context_text(self, context: List[Dict] = None) -> str:
        """
        构建上下文文本 — 支持窗口管理

        策略：
        - 合并传入的 context 和内部 _context_history
        - 保留最近 max_context_turns 轮
        - 每轮格式：用户: xxx / 助手: xxx

        Args:
            context: 外部传入的对话历史

        Returns:
            格式化后的上下文文本
        """
        # 合并历史
        all_history = list(self._context_history)
        if context:
            all_history.extend(context)

        # 截断到最大轮数
        if len(all_history) > self.max_context_turns:
            all_history = all_history[-self.max_context_turns:]

        if not all_history:
            return "(无对话历史)"

        lines = []
        for msg in all_history:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "user":
                lines.append(f"用户: {content}")
            elif role == "assistant":
                lines.append(f"助手: {content}")
            elif role == "system":
                lines.append(f"系统: {content}")

        return "\n".join(lines)

    # ============================================================
    #  阶段 1: 规划
    # ============================================================

    def _plan(self, task: str, state: AgentState, context: List[Dict] = None) -> Optional[Plan]:
        """初始规划"""
        context_text = self._build_context_text(context)
        prompt = self.PLANNING_PROMPT.format(
            business_context=self.business_context,
            tools=ToolRegistry.get_tool_descriptions(),
            task=task,
            context=context_text,
        )
        response = self._call_llm(prompt, state)
        if not response:
            return None

        subtasks = self._parse_subtasks(response["content"])
        if not subtasks:
            # 如果解析失败，回退到单任务
            subtasks = [SubTask(
                id="task_1",
                description=task,
                depends_on=[],
                tool="none",
                tool_input="直接回答",
            )]

        plan = Plan(
            original_task=task,
            subtasks=subtasks,
        )
        plan.execution_order = self._topological_sort(subtasks)
        plan.parallel_groups = self._find_parallel_groups(subtasks)

        return plan

    def _plan_with_context(self, task: str, old_plan: Plan,
                           reflection: Dict, state: AgentState,
                           context: List[Dict] = None) -> Optional[Plan]:
        """基于反思结果重新规划"""
        # 构建已有结果上下文
        completed = []
        for s in old_plan.subtasks:
            if s.status in ("success", "failed"):
                completed.append(f"- {s.id}: {s.description} → {s.status} → {s.result[:100]}")

        context_text = self._build_context_text(context)
        prompt = self.PLANNING_PROMPT.format(
            business_context=self.business_context,
            tools=ToolRegistry.get_tool_descriptions(),
            task=task,
            context=context_text,
        )
        prompt += f"\n\n已完成子任务：\n{''.join(completed)}"
        prompt += f"\n反思建议：{reflection.get('adjustments', '继续执行')}"
        prompt += "\n请输出更新后的规划（只包含尚未完成或需要修改的子任务）："

        response = self._call_llm(prompt, state)
        if not response:
            return None

        new_subtasks = self._parse_subtasks(response["content"])
        if not new_subtasks:
            return old_plan

        # 合并：保留已成功的子任务，加入新子任务
        success_ids = {s.id for s in old_plan.subtasks if s.status == "success"}
        plan = Plan(
            original_task=task,
            subtasks=old_plan.subtasks + new_subtasks,
        )
        plan.execution_order = self._topological_sort(plan.subtasks)
        plan.parallel_groups = self._find_parallel_groups(plan.subtasks)

        return plan

    # ============================================================
    #  阶段 2: 执行
    # ============================================================

    def _execute_plan(self, plan: Plan, state: AgentState):
        """按执行顺序执行子任务"""
        completed_ids = set()

        for task_id in plan.execution_order:
            subtask = self._find_subtask(plan.subtasks, task_id)
            if not subtask:
                continue

            # 跳过已成功的
            if subtask.status == "success":
                completed_ids.add(task_id)
                continue

            # 检查依赖是否满足
            deps_met = all(d in completed_ids for d in subtask.depends_on)
            if not deps_met:
                subtask.status = "skipped"
                subtask.result = "依赖子任务未完成"
                continue

            # 执行
            subtask.status = "running"
            success = self._execute_subtask(subtask, state)

            if success:
                subtask.status = "success"
                completed_ids.add(task_id)
                state.subtask_results[task_id] = subtask.result
            else:
                # 重试
                for attempt in range(subtask.max_retries - 1):
                    subtask.retry_count += 1
                    if self._execute_subtask(subtask, state):
                        subtask.status = "success"
                        completed_ids.add(task_id)
                        state.subtask_results[task_id] = subtask.result
                        break
                else:
                    subtask.status = "failed"

    def _execute_subtask(self, subtask: SubTask, state: AgentState) -> bool:
        """执行单个子任务"""
        if subtask.tool == "none":
            # 不需要工具，直接调用 LLM（带上下文）
            context_text = self._build_context_text()
            prompt = f"""请完成以下任务：
{subtask.description}

对话历史：
{context_text}

直接回答："""
            response = self._call_llm(prompt, state)
            if response:
                subtask.result = response["content"]
                return True
            subtask.result = f"错误: LLM 调用失败 — {self._last_llm_error or '未知原因'}"
            return False

        if not self.enable_tools:
            subtask.result = "工具未启用"
            return False

        result = ToolRegistry.execute(subtask.tool, subtask.tool_input, state)
        subtask.result = result

        # 判断是否成功（简单启发式）
        return not result.startswith("错误:")

    # ============================================================
    #  阶段 2: 反思
    # ============================================================

    def _reflect(self, task: str, plan: Plan, state: AgentState) -> Dict:
        """反思当前进度"""
        # 构建已完成结果
        results_lines = []
        for s in plan.subtasks:
            results_lines.append(f"- {s.id} ({s.status}): {s.description}")
            if s.result:
                results_lines.append(f"  结果: {s.result[:200]}")
        results_text = "\n".join(results_lines)

        prompt = self.REFLECTION_PROMPT.format(
            task=task,
            results=results_text,
        )

        response = self._call_llm(prompt, state)
        if not response:
            return {"status": "continue", "reason": "LLM 调用失败，默认继续", "adjustments": ""}

        # 解析 JSON
        try:
            json_str = self._extract_json(response["content"])
            return json.loads(json_str)
        except:
            return {"status": "continue", "reason": "解析失败，默认继续", "adjustments": ""}

    # ============================================================
    #  阶段 3: 总结
    # ============================================================

    def _summarize(self, task: str, plan: Plan, state: AgentState) -> str:
        """汇总所有子任务结果"""
        results_lines = []
        for s in plan.subtasks:
            results_lines.append(f"### {s.id}: {s.description}")
            results_lines.append(f"状态: {s.status}")
            if s.result:
                results_lines.append(f"结果: {s.result}")
            results_lines.append("")

        context_text = self._build_context_text()
        prompt = self.SUMMARY_PROMPT.format(
            task=task,
            results="\n".join(results_lines),
        )
        prompt += f"\n\n对话历史：\n{context_text}"

        response = self._call_llm(prompt, state)
        return response["content"] if response else "(总结生成失败)"

    # ============================================================
    #  内部方法
    # ============================================================

    def _call_llm(self, prompt: str, state: AgentState) -> Optional[Dict]:
        """调用 LLM"""
        self._last_llm_error = None
        if not isinstance(prompt, str):
            self._last_llm_error = "内部错误: prompt 类型无效"
            return None
        if len(prompt) > _MAX_TASK_LEN * 2:
            self._last_llm_error = f"内部错误: prompt 过长（>{_MAX_TASK_LEN * 2}）"
            return None
        if not self.api_key:
            self._last_llm_error = "未配置 API Key（请设置 DASHSCOPE_API_KEY 或传入 api_key）"
            return None
        try:
            state.total_llm_calls += 1
            messages = [{"role": "user", "content": prompt}]
            response = requests.post(
                f"{self.api_base}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": self.llm_max_tokens,
                    "temperature": self.temperature,
                },
                timeout=self.llm_timeout,
            )
            if response.status_code == 200:
                result = response.json()
                try:
                    content = result["choices"][0]["message"]["content"]
                except (KeyError, IndexError, TypeError) as e:
                    self._last_llm_error = f"响应 JSON 结构异常: {e}; 原始片段: {json.dumps(result, ensure_ascii=False)[:600]}"
                    return None
                tokens = result.get("usage", {}).get("total_tokens", 0)
                state.total_tokens += tokens
                return {"content": content, "tokens": tokens}
            self._last_llm_error = _http_error_detail(response)
            return None
        except requests.Timeout:
            self._last_llm_error = f"请求超时（{self.llm_timeout} 秒）"
            return None
        except requests.RequestException as e:
            self._last_llm_error = f"网络请求异常: {type(e).__name__}: {e}"
            return None
        except Exception as e:
            self._last_llm_error = f"处理 LLM 响应失败: {type(e).__name__}: {e}"
            return None

    def _parse_subtasks(self, content: str) -> List[SubTask]:
        """从 LLM 输出中解析子任务列表"""
        try:
            json_str = self._extract_json(content)
            data = json.loads(json_str)
            if isinstance(data, list):
                return [
                    SubTask(
                        id=item.get("id", f"task_{i+1}"),
                        description=item.get("description", ""),
                        depends_on=item.get("depends_on", []),
                        tool=item.get("tool", "none"),
                        tool_input=item.get("tool_input", ""),
                    )
                    for i, item in enumerate(data)
                ]
        except:
            pass
        return []

    def _extract_json(self, text: str) -> str:
        """从文本中提取 JSON"""
        # 尝试提取 ```json ... ``` 块
        match = re.search(r'```(?:json)?\s*\n(.*?)\n```', text, re.DOTALL)
        if match:
            return match.group(1).strip()
        # 尝试提取 [...] 或 {...}
        match = re.search(r'(\[.*\]|\{.*\})', text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return text.strip()

    def _topological_sort(self, subtasks: List[SubTask]) -> List[str]:
        """拓扑排序 — 确定执行顺序"""
        task_map = {s.id: s for s in subtasks}
        visited = set()
        order = []

        def visit(task_id):
            if task_id in visited:
                return
            visited.add(task_id)
            task = task_map.get(task_id)
            if task:
                for dep in task.depends_on:
                    visit(dep)
                order.append(task_id)

        for s in subtasks:
            visit(s.id)

        return order

    def _find_parallel_groups(self, subtasks: List[SubTask]) -> List[List[str]]:
        """找并行组 — 无依赖关系的子任务可并行"""
        task_map = {s.id: s for s in subtasks}
        completed = set()
        groups = []

        remaining = list(subtasks)
        while remaining:
            # 找到所有依赖已满足的子任务
            ready = [
                s for s in remaining
                if all(d in completed for d in s.depends_on)
            ]
            if not ready:
                break
            groups.append([s.id for s in ready])
            for s in ready:
                completed.add(s.id)
                remaining.remove(s)

        return groups

    def _find_subtask(self, subtasks: List[SubTask], task_id: str) -> Optional[SubTask]:
        for s in subtasks:
            if s.id == task_id:
                return s
        return None

    def _build_summary(self, state: AgentState, elapsed: float) -> str:
        """构建执行摘要"""
        plan = state.plan
        lines = [
            f"=== 规划-执行-反思 智能体执行摘要 ===",
            f"任务: {state.original_task[:100]}",
            f"LLM 调用: {state.total_llm_calls} 次",
            f"Token 消耗: {state.total_tokens}",
            f"重新规划: {state.replan_count} 次",
            f"耗时: {elapsed:.2f}s",
            f"子任务: {sum(1 for s in plan.subtasks if s.status == 'success')}/{len(plan.subtasks)} 成功",
        ]

        if state.reflection_log:
            lines.append(f"\n反思记录:")
            for log in state.reflection_log:
                lines.append(f"  - {log}")

        lines.append(f"\n子任务详情:")
        for s in plan.subtasks:
            status_icon = {"success": "✅", "failed": "❌", "skipped": "⏭️"}.get(s.status, "⏳")
            lines.append(f"  {status_icon} {s.id}: {s.description[:60]}")
            if s.result:
                lines.append(f"     结果: {s.result[:150]}")
            if s.retry_count > 0:
                lines.append(f"     重试: {s.retry_count} 次")

        if state.final_answer:
            lines.append(f"\n最终答案: {state.final_answer}")

        return "\n".join(lines)


# ============================================================
#  测试
# ============================================================

if __name__ == "__main__":
    agent = CustomAgent()

    print("=" * 60)
    print("测试 1: 多步骤任务")
    print("=" * 60)
    result = agent.run("计算 25 * 4 + 100 / 5 的结果，然后把结果存储到记忆中")
    print(f"成功: {result['success']}")
    print(f"输出: {result['output'][:500]}")
    if "_meta" in result:
        m = result["_meta"]
        print(f"子任务: {m['subtasks_success']}/{m['subtasks_total']} 成功")
        print(f"LLM 调用: {m['llm_calls']} 次, Token: {m['tokens']}")

    print("\n" + "=" * 60)
    print("测试 2: 代码执行任务")
    print("=" * 60)
    agent.reset()
    result = agent.run("用 Python 计算斐波那契数列前 20 项，并找出其中的偶数")
    print(f"成功: {result['success']}")
    print(f"输出: {result['output'][:500]}")

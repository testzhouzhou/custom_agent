#!/usr/bin/env python3
"""
自定义工具示例

每个工具是一个函数，接收参数，返回结果。
智能体可以通过工具调用实现更复杂的功能。
"""


def search_web(query: str) -> str:
    """搜索网页"""
    # 实现你的搜索逻辑
    return f"搜索结果: {query}"


def read_file(path: str) -> str:
    """读取文件"""
    try:
        with open(path, "r") as f:
            return f.read()
    except Exception as e:
        return f"读取失败: {e}"


def execute_command(cmd: str) -> str:
    """执行命令"""
    import subprocess
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
        return result.stdout
    except Exception as e:
        return f"执行失败: {e}"

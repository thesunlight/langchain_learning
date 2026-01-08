"""
LangChain 学习项目 - 实用工具函数

本文件包含一些常用的工具函数和辅助功能，用于支持 LangChain 应用开发。
"""

import os
from typing import Dict, List, Any
import json
from datetime import datetime
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

def setup_environment():
    """
    设置运行环境，检查必要的环境变量
    """
    print("=== 设置运行环境 ===")
    
    # 检查必要的环境变量
    required_vars = ['OPENAI_API_KEY']
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"警告: 缺少以下环境变量: {', '.join(missing_vars)}")
        print("请在 .env 文件中设置这些变量")
        print("示例 .env 文件内容:")
        print("OPENAI_API_KEY=your_openai_api_key_here")
    else:
        print("所有必需的环境变量都已设置")
    
    print()

def print_model_info():
    """
    打印模型相关信息
    """
    print("=== LangChain 模型信息 ===")
    print("LangChain 支持多种语言模型:")
    print("1. OpenAI 模型 (GPT-3.5, GPT-4 等)")
    print("2. Anthropic 模型 (Claude 系列)")
    print("3. Google 模型 (PaLM, Gemini 等)")
    print("4. 本地模型 (通过 Hugging Face 等)")
    print("5. 自定义模型接口")
    print()

def format_prompt_example():
    """
    格式化提示词的示例
    """
    print("=== 提示词格式化示例 ===")
    
    # 示例模板
    template = """
    任务: {task}
    
    上下文:
    {context}
    
    指令:
    {instructions}
    
    请根据以上信息完成任务。
    """
    
    formatted_prompt = template.format(
        task="撰写产品描述",
        context="这是一款面向年轻人的智能手表",
        instructions="重点突出健康监测功能和时尚外观"
    )
    
    print("格式化后的提示词:")
    print(formatted_prompt.strip())
    print()

def safe_eval(expression: str) -> Any:
    """
    安全的表达式求值函数
    
    Args:
        expression: 要计算的表达式
        
    Returns:
        计算结果或错误信息
    """
    try:
        # 定义安全的命名空间
        safe_dict = {
            "__builtins__": {},
            "abs": abs,
            "round": round,
            "min": min,
            "max": max,
            "sum": sum,
            "pow": pow,
            "divmod": divmod,
            "complex": complex,
            "float": float,
            "int": int,
            "math": __import__('math')
        }
        
        # 安全计算表达式
        result = eval(expression, {"__builtins__": None}, safe_dict)
        return result
    except Exception as e:
        return f"计算错误: {str(e)}"

def validate_json(json_string: str) -> bool:
    """
    验证JSON字符串的有效性
    
    Args:
        json_string: JSON字符串
        
    Returns:
        是否为有效JSON
    """
    try:
        json.loads(json_string)
        return True
    except json.JSONDecodeError:
        return False

def get_current_datetime() -> str:
    """
    获取当前日期和时间的格式化字符串
    
    Returns:
        格式化的日期时间字符串
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def calculate_levenshtein_distance(s1: str, s2: str) -> int:
    """
    计算两个字符串之间的编辑距离（Levenshtein距离）
    
    Args:
        s1: 第一个字符串
        s2: 第二个字符串
        
    Returns:
        编辑距离
    """
    if len(s1) < len(s2):
        return calculate_levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]

def similarity_score(text1: str, text2: str) -> float:
    """
    计算两个文本的相似度分数（0-1之间）
    
    Args:
        text1: 第一个文本
        text2: 第二个文本
        
    Returns:
        相似度分数
    """
    distance = calculate_levenshtein_distance(text1, text2)
    max_len = max(len(text1), len(text2))
    if max_len == 0:
        return 1.0
    return 1 - (distance / max_len)

def batch_process(items: List[Any], process_func, batch_size: int = 5) -> List[Any]:
    """
    批量处理项目列表
    
    Args:
        items: 要处理的项目列表
        process_func: 处理函数
        batch_size: 批次大小
        
    Returns:
        处理结果列表
    """
    results = []
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        batch_results = [process_func(item) for item in batch]
        results.extend(batch_results)
    return results

def format_large_number(num: float) -> str:
    """
    将大数字格式化为更易读的形式
    
    Args:
        num: 数字
        
    Returns:
        格式化后的字符串
    """
    if num >= 1_000_000_000:
        return f"{num / 1_000_000_000:.1f}B"
    elif num >= 1_000_000:
        return f"{num / 1_000_000:.1f}M"
    elif num >= 1_000:
        return f"{num / 1_000:.1f}K"
    else:
        return str(num)

def clean_text(text: str) -> str:
    """
    清理文本，移除多余的空白字符
    
    Args:
        text: 原始文本
        
    Returns:
        清理后的文本
    """
    import re
    # 移除多余的空白字符，保留句子结构
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

if __name__ == "__main__":
    print("LangChain 实用工具函数")
    print("="*50)
    
    setup_environment()
    print_model_info()
    format_prompt_example()
    
    print("=== 工具函数测试 ===")
    
    # 测试安全计算
    print(f"安全计算 '2 + 3 * 4': {safe_eval('2 + 3 * 4')}")
    
    # 测试JSON验证
    print(f"JSON验证 {{\"key\": \"value\"}}: {validate_json('{\"key\": \"value\"}')}")
    print(f"JSON验证 invalid json: {validate_json('invalid json')}")
    
    # 测试日期时间
    print(f"当前时间: {get_current_datetime()}")
    
    # 测试相似度
    sim_score = similarity_score("hello world", "hello word")
    print(f"文本相似度 ('hello world', 'hello word'): {sim_score:.2f}")
    
    # 测试大数字格式化
    print(f"大数字格式化 1500000: {format_large_number(1500000)}")
    
    # 测试文本清理
    dirty_text = "  这是   一个包含多余空格  的文本  "
    print(f"文本清理: '{clean_text(dirty_text)}'")
    
    print("\n所有工具函数测试完成！")